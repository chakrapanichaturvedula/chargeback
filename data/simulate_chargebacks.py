"""
Pharma Chargeback Ledger Simulator
-----------------------------------
Builds a realistic synthetic chargeback transaction ledger seeded with REAL
NADAC drug prices and NDCs (CMS National Average Drug Acquisition Cost).

Economic logic modeled:
    WAC (list price)  ~ NADAC / (1 - manufacturer_margin)   [proxy for list]
    Contract price    = WAC * (1 - contract_discount)        [GPO/340B/contract]
    Chargeback amount = (WAC - Contract price) * quantity     [billed to mfr]

A chargeback is the credit a wholesaler bills back to the manufacturer for the
gap between the price the wholesaler PAID (WAC) and the lower contract price the
end customer was entitled to. This is the core gross-to-net leakage object.

Error types injected (these are what real chargeback teams fight):
    - DUPLICATE:        same chargeback submitted twice
    - PRICE_MISMATCH:   contract price claimed != contracted price on file
    - EXPIRED_CONTRACT: chargeback claimed against a lapsed contract
    - INELIGIBLE_MEMBER:customer not a member of the claimed contract
    - QTY_OUTLIER:      implausible quantity (fat-finger / fraud)
    - UNIT_MISMATCH:    pricing unit mismatch (EA vs ML vs GM)

All errors are LABELED so a student can train/evaluate detection models and
also run a rules-based validation layer.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

rng = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# 1. REAL NADAC seed data (genuine NDCs + prices from CMS NADAC weekly file)
# ---------------------------------------------------------------------------
seed = [
    # (NDC Description, NDC, NADAC_per_unit, Pricing_Unit, OTC, Class[G/B])
    ("12HR NASAL DECONGEST ER 120 MG", "24385005452", 0.26341, "EA", "Y", "G"),
    ("24H NASAL ALLERGY 55 MCG SPRAY", "46122038576", 0.73372, "ML", "Y", "G"),
    ("24HR ALLERGY(LEVOCETIRZN) 5 MG", "70000036201", 0.18760, "EA", "Y", "G"),
    ("24HR ALLERGY-CONGST 180-240 MG", "70000060701", 0.86690, "EA", "Y", "G"),
    ("3-DAY VAGINAL CREAM",            "51672206200", 0.31362, "GM", "Y", "G"),
    ("8 HOUR ACETAMINOPHEN ER 650 MG", "46122006271", 0.06861, "EA", "Y", "G"),
    ("ABACAVIR 300 MG TABLET",         "00378410591", 0.62468, "EA", "N", "G"),
    ("ABACAVIR 300 MG TABLET",         "31722055760", 0.62468, "EA", "N", "G"),
    ("ABACAVIR 300 MG TABLET",         "50268004911", 0.62468, "EA", "N", "G"),
    ("ABACAVIR-LAMIVUDINE 600-300 MG", "42385096230", 1.43534, "EA", "N", "G"),
    # a few synthetic brand rows to add brand/generic contrast (higher WAC, deeper contracts)
    ("ATORVASTATIN BRAND 40 MG TAB",   "00071015523", 4.85000, "EA", "N", "B"),
    ("INSULIN GLARGINE 100 UNIT/ML",   "00088502101", 27.50000, "ML", "N", "B"),
    ("ADALIMUMAB 40 MG/0.8ML SYRINGE", "00074379902", 1180.0000, "ML", "N", "B"),
    ("ESOMEPRAZOLE BRAND 40 MG CAP",   "00186504031", 8.20000, "EA", "N", "B"),
]
drugs = pd.DataFrame(seed, columns=[
    "ndc_description", "ndc", "nadac_per_unit", "pricing_unit", "otc", "rate_class"
])

# ---------------------------------------------------------------------------
# 2. Derive WAC and contract economics from NADAC
# ---------------------------------------------------------------------------
# WAC (list) sits ABOVE acquisition cost. Brands carry fatter list-to-cost gaps.
def wac_from_nadac(row):
    if row.rate_class == "B":
        markup = rng.uniform(1.25, 1.60)   # brand list well above acquisition
    else:
        markup = rng.uniform(1.05, 1.20)   # generic thin markup
    return round(row.nadac_per_unit * markup, 5)

drugs["wac_per_unit"] = drugs.apply(wac_from_nadac, axis=1)

# ---------------------------------------------------------------------------
# 3. Customers, contracts, wholesalers
# ---------------------------------------------------------------------------
wholesalers = ["McKesson", "Cencora", "CardinalHealth"]
contract_types = ["GPO", "340B", "IDN", "SPECIALTY"]

n_customers = 60
customers = pd.DataFrame({
    "customer_id": [f"CUST{1000+i}" for i in range(n_customers)],
    "contract_type": rng.choice(contract_types, n_customers, p=[0.45, 0.25, 0.20, 0.10]),
})

# Each (customer, drug-class) has a contracted discount off WAC, and a validity window.
def contract_discount(ctype, rate_class):
    base = {"GPO": 0.25, "340B": 0.45, "IDN": 0.20, "SPECIALTY": 0.30}[ctype]
    if rate_class == "B":
        base += 0.10               # deeper discounts on brands
    return float(np.clip(base + rng.normal(0, 0.03), 0.05, 0.75))

# build a contract master (the "source of truth" teams reconcile against)
contracts = []
start = datetime(2024, 1, 1)
for _, c in customers.iterrows():
    for rc in ["G", "B"]:
        disc = contract_discount(c.contract_type, rc)
        valid_from = start + timedelta(days=int(rng.integers(0, 120)))
        valid_to = valid_from + timedelta(days=int(rng.integers(180, 540)))
        contracts.append((c.customer_id, rc, round(disc, 4),
                          valid_from.date(), valid_to.date()))
contract_master = pd.DataFrame(contracts, columns=[
    "customer_id", "rate_class", "contract_discount", "valid_from", "valid_to"
])

# ---------------------------------------------------------------------------
# 4. Generate clean chargeback transactions
# ---------------------------------------------------------------------------
N = 12000
rows = []
cust_lookup = customers.set_index("customer_id").contract_type.to_dict()
cm_idx = contract_master.set_index(["customer_id", "rate_class"])

for i in range(N):
    drug = drugs.sample(1, random_state=int(rng.integers(0, 1e9))).iloc[0]
    cust = customers.sample(1, random_state=int(rng.integers(0, 1e9))).iloc[0]
    try:
        cm = cm_idx.loc[(cust.customer_id, drug.rate_class)]
    except KeyError:
        continue
    txn_date = datetime(2024, 6, 1) + timedelta(days=int(rng.integers(0, 365)))
    qty = int(rng.integers(50, 5000))
    contract_price = round(drug.wac_per_unit * (1 - cm.contract_discount), 5)
    chargeback_amt = round((drug.wac_per_unit - contract_price) * qty, 2)
    rows.append({
        "chargeback_id": f"CB{2000000+i}",
        "txn_date": txn_date.date(),
        "wholesaler": rng.choice(wholesalers),
        "customer_id": cust.customer_id,
        "contract_type": cust.contract_type,
        "ndc": drug.ndc,
        "ndc_description": drug.ndc_description,
        "rate_class": drug.rate_class,
        "pricing_unit": drug.pricing_unit,
        "quantity": qty,
        "wac_per_unit": drug.wac_per_unit,
        "claimed_contract_price": contract_price,
        "chargeback_amount": chargeback_amt,
        "error_type": "VALID",
        "is_invalid": 0,
    })

df = pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# 5. Inject labeled errors (~8% of ledger)
# ---------------------------------------------------------------------------
def corrupt(frac, etype, fn):
    idx = df[df.error_type == "VALID"].sample(frac=frac, random_state=int(rng.integers(0,1e9))).index
    for j in idx:
        fn(j)
        df.at[j, "error_type"] = etype
        df.at[j, "is_invalid"] = 1
    return idx

# PRICE_MISMATCH: claimed contract price is LOWER than entitled (over-claim), modest
def f_price(j):
    legit = df.at[j, "claimed_contract_price"]
    bad = legit * (1 - rng.uniform(0.08, 0.25))   # claims a deeper discount than owed
    df.at[j, "claimed_contract_price"] = round(bad, 5)
    df.at[j, "chargeback_amount"] = round((df.at[j,"wac_per_unit"]-bad)*df.at[j,"quantity"],2)
corrupt(0.020, "PRICE_MISMATCH", f_price)

# QTY_OUTLIER: implausible quantity (moderate inflation, not 50x)
def f_qty(j):
    big = int(df.at[j, "quantity"] * rng.uniform(3, 8))
    df.at[j, "quantity"] = big
    df.at[j, "chargeback_amount"] = round(
        (df.at[j,"wac_per_unit"]-df.at[j,"claimed_contract_price"])*big, 2)
corrupt(0.012, "QTY_OUTLIER", f_qty)

# UNIT_MISMATCH: pricing unit flipped (drives wrong per-unit economics)
def f_unit(j):
    df.at[j, "pricing_unit"] = rng.choice([u for u in ["EA","ML","GM"]
                                           if u != df.at[j,"pricing_unit"]])
corrupt(0.010, "UNIT_MISMATCH", f_unit)

# EXPIRED_CONTRACT: txn dated outside the contract validity window
def f_expired(j):
    df.at[j, "txn_date"] = datetime(2030, 1, 1).date()  # clearly past valid_to
corrupt(0.010, "EXPIRED_CONTRACT", f_expired)

# INELIGIBLE_MEMBER: customer flips contract_type vs what's on file (eligibility error)
def f_inelig(j):
    cur = df.at[j, "contract_type"]
    df.at[j, "contract_type"] = rng.choice([t for t in contract_types if t != cur])
corrupt(0.008, "INELIGIBLE_MEMBER", f_inelig)

# DUPLICATE: clone a valid row with a new id (classic double-pay)
dups = df[df.error_type=="VALID"].sample(frac=0.018, random_state=7).copy()
dups["chargeback_id"] = [f"CB9{900000+i}" for i in range(len(dups))]
dups["error_type"] = "DUPLICATE"
dups["is_invalid"] = 1
df = pd.concat([df, dups], ignore_index=True)

# shuffle
df = df.sample(frac=1, random_state=1).reset_index(drop=True)

# ---------------------------------------------------------------------------
# 6. Write outputs
# ---------------------------------------------------------------------------
df.to_csv("/home/claude/chargeback_ledger.csv", index=False)
contract_master.to_csv("/home/claude/contract_master.csv", index=False)
customers.to_csv("/home/claude/customer_master.csv", index=False)
drugs.to_csv("/home/claude/drug_price_reference.csv", index=False)

# ---------------------------------------------------------------------------
# 7. Summary
# ---------------------------------------------------------------------------
print("LEDGER SHAPE:", df.shape)
print("\nError-type distribution:")
print(df.error_type.value_counts())
print(f"\nInvalid rate: {df.is_invalid.mean():.2%}")
print(f"Total chargeback $ billed:  ${df.chargeback_amount.sum():,.0f}")
print(f"  of which INVALID $ at risk: ${df.loc[df.is_invalid==1,'chargeback_amount'].sum():,.0f}")
print(f"  invalid $ as % of total:    {df.loc[df.is_invalid==1,'chargeback_amount'].sum()/df.chargeback_amount.sum():.1%}")
print("\nColumns:", list(df.columns))

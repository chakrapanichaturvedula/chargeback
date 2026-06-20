# Chargeback Integrity Analytics — Applied Case: Shanta Pharma Limited

A portfolio / interview project on **pharmaceutical chargeback integrity**: detecting invalid, duplicated, and mispriced chargeback claims in a manufacturer's gross-to-net workflow, and quantifying the dollars at risk.

**Live site:** once Pages is enabled → `https://chakrapanichaturvedula.github.io/chargeback/`

---

## What's here

| File | Description |
|------|-------------|
| `index.html` | The project website (single page, no build step) |
| `data/chargeback_ledger.csv` | Main table — 12,203 labelled chargeback lines |
| `data/contract_master.csv` | Contracted discounts + validity windows (reconciliation source of truth) |
| `data/customer_master.csv` | 60 customers and their contract channels |
| `data/drug_price_reference.csv` | NDC-level NADAC / WAC price spine (14 products) |
| `data/simulate_chargebacks.py` | The data generator — tune parameters and reproduce |
| `data/Shanta_Pharma_Chargeback_Project.docx` | Problem statement + data dictionary |

## The problem in one line

`chargeback = (WAC − contract price) × quantity`. Manufacturers pay these back to wholesalers at huge volume; a share are wrong. This project finds the wrong ones.

## Dataset at a glance

- 12,203 chargeback lines across 60 customers
- ~$1.61B total billed; ~7.4% of lines flagged; ~$192.7M (12%) value at risk
- Six labelled error types: duplicate, price mismatch, expired contract, ineligible member, quantity outlier, unit mismatch

## Honest note

Academic exercise. **"Shanta Pharma Limited" is illustrative, not a real company.** Drug codes and acquisition costs come from the public CMS NADAC file; contract terms, error rates, and the error mix are modelling assumptions, not measured industry figures. The generator is included so every number is reproducible.

---

## Deploy to GitHub Pages (2 minutes)

1. Create the repo (if not already): `https://github.com/chakrapanichaturvedula/chargeback`
2. Copy everything from this folder into the repo root (so `index.html` sits at the top level and `data/` is alongside it).
3. Commit and push:
   ```bash
   git add .
   git commit -m "Chargeback integrity analytics site + datasets"
   git push origin main
   ```
4. On GitHub: **Settings → Pages → Source: Deploy from a branch → Branch: `main` / root → Save.**
5. Wait ~1 minute. Your site is live at `https://chakrapanichaturvedula.github.io/chargeback/`.

All download links are relative (`data/...`), so they work the moment Pages is live.

## Regenerate the data

```bash
cd data
pip install pandas numpy
python simulate_chargebacks.py
```

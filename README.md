# Floodwall Design Tool

Streamlit web app for preliminary design of reinforced concrete floodwalls (cantilever and counterfort): wall sizing, stability checks, RCC detailing and scour depth.

## What it does
1. **Sizing and optimization**: searches for the smallest safe cantilever section (base width, toe and heel lengths, slab and stem thickness, optional shear key). Walls taller than 7 m are sized as counterfort walls.
2. **Stability checks**: overturning, sliding, bearing pressure and eccentricity, for static and seismic cases.
3. **RCC design**: stem, base slab and shear key reinforcement, with checks for shear, crack width and development length. If a check fails, slab and stem thickness are increased in 50 mm steps and the design is rerun (up to 20 iterations).
4. **Scour analysis**: required foundation depth from four methods, plus riprap and geotextile recommendations.
5. **Batch processing**: designs many cross-sections from an Excel or CSV file and exports the results to Excel.

## Design basis
| Item | Basis |
|---|---|
| Concrete and steel | IS 456:2000 (limit state design), fck 20-35 MPa, fy 415-550 MPa |
| Stability criteria | IS 3370:2009 |
| Seismic loads | NBC 105:2020 / IS 1893:2016 |
| Surcharge | IRC 6:2017 equivalent surcharge height |
| Static earth pressure | Rankine (Ka, Kp, K0) |
| Seismic earth pressure | Mononobe-Okabe (KAE) |
| Hydraulics | Flood-side hydrostatic force, base uplift with a drain/cutoff relief factor, optional protected-side water level |
| Scour | Lacey regime depth, degradation with margin, modified Lacey, USACE guideline, practical minimum |

**Required factors of safety**

| Check | Static | Seismic |
|---|---|---|
| Overturning | 2.0 | 1.5 |
| Sliding | 1.5 | 1.25 |
| Bearing pressure | ≤ safe bearing capacity | ≤ safe bearing capacity |
| Eccentricity | ≤ B/6 | ≤ B/4 |

## Inputs (sidebar)
- **Site and hydraulics:** high flood level, ground level, protected-side water level, freeboard, design discharge, river width, silt factor, D50, drain/cutoff relief factor
- **Soil:** friction angle, unit weight, base friction coefficient, safe bearing capacity, foundation depth, surcharge height, shear key mode (Auto / Disabled / Enabled)
- **Materials:** concrete grade, steel grade
- **Seismic:** PGA (Z), importance factor, site factor, response factor

## Run locally
```
pip install -r requirements.txt
streamlit run floodwall_design_app.py
```

## Batch input format
An Excel or CSV file with these columns:

| Station | HFL | Ground_Level |
|---|---|---|
| 0+000 | 105.50 | 100.00 |

## Limitations
- Preliminary sizing only, not a substitute for detailed design.
- For counterfort walls the app does the sizing and stability checks, but the RCC module uses cantilever-stem formulas, so counterfort and panel reinforcement is not designed.
- Static earth pressure assumes level backfill and a single uniform soil layer.
- Seepage uplift is a simplified linear model controlled by the relief factor.
- Candidate sections are ranked using a simplified unit-rate cost estimate.
- Default input values are illustrative examples.

## Disclaimer
For learning and preliminary design only. Results must be checked by a qualified engineer before use in any design for construction.

## Author
Ujjwal Lohani

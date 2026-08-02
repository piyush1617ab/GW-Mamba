# Data

The PEMS03 / PEMS04 / PEMS07 / PEMS08 traffic datasets are not committed to this repository due to size.

## Obtaining the data

These are the standard PEMS benchmark files used throughout the traffic forecasting literature (as used by Graph WaveNet, DCRNN, STAEformer, and related work). They are commonly distributed as `.npz` files (traffic flow tensor) plus a distance/adjacency file per dataset.

| Dataset | Sensors | Timesteps | Channels | Period |
|---|---|---|---|---|
| PEMS03 | 358 | 26,208 | 1 | 6 months |
| PEMS04 | 307 | 16,992 | 3 | Jan–Feb 2018 |
| PEMS07 | 883 | 28,224 | 1 | May–Aug 2017 |
| PEMS08 | 170 | 17,856 | 3 | Jul–Aug 2016 |

## Expected structure

Place downloaded files as:

```
data/
├── PEMS03/
│   ├── PEMS03.npz
│   └── PEMS03.csv
├── PEMS04/
│   └── ...
├── PEMS07/
│   └── ...
└── PEMS08/
    └── ...
```

The notebooks in `notebooks/final/` expect this layout. Update the data path at the top of each notebook if your local structure differs.

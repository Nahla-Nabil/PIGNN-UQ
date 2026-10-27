<div align="center">

# PIGNN-UQ

### Physics-Informed Graph Neural Networks with Uncertainty Quantification for Registry-Compatible Blue Carbon Crediting

**Nahla Nabil** · Faculty of Computer Studies, Arab Open University, Kingdom of Bahrain

[![Conference](https://img.shields.io/badge/ICDABI%202026-DATA'26-3B6FB6?style=for-the-badge)](#)
[![IEEE](https://img.shields.io/badge/IEEE-Conference%20Paper-E8749A?style=for-the-badge)](#)
[![Paper](https://img.shields.io/badge/Paper-PDF-F2C14E?style=for-the-badge)](paper/main.pdf)
[![Solo](https://img.shields.io/badge/Research-Single%20author-5BAA7A?style=for-the-badge)](#)

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.13-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyTorch%20Geometric-2.8-3C2179)](https://pyg.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.4-5BAA7A)](https://xgboost.ai/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Data](https://img.shields.io/badge/Data-Sentinel--2%20%7C%203DEP%20%7C%20NOAA-9CC3E6)](#data-sources)
[![LaTeX](https://img.shields.io/badge/Paper-LaTeX%20IEEEtran-008080?logo=latex&logoColor=white)](paper/main.tex)

</div>

---

> **The problem.** Mangrove carbon credits are discounted by a non-permanence buffer that does
> not depend on how uncertain the underlying carbon-stock estimate is.
>
> **What PIGNN-UQ does.** It maps mangrove soil carbon on a spatial graph, combines a
> physics-informed graph neural network with XGBoost, attaches a 90% conformal interval
> (calibrated on held-out map labels) to every 150 m node, and converts that interval into the
> uncertainty deduction of **Verra VM0033 v2.1** (p. 64, Eq. 92), kept separate from the
> registry's non-permanence buffer.

<p align="center">
  <img src="paper/figures/rev_fig0_framework.png" alt="PIGNN-UQ pipeline" width="560">
</p>

## Contents

| Start here | Method | Results | Use the code |
|---|---|---|---|
| [Key results](#key-results) | [Pipeline](#pipeline) | [Accuracy](#accuracy) | [Reproduce](#reproducing-the-results) |
| [Paper files](#paper-files) | [Data](#study-regions-and-data) | [Intervals and crediting](#intervals-and-crediting) | [Repository layout](#repository-layout) |
| [Tech stack](#tech-stack) | [Evaluation protocol](#evaluation-protocol) | [Field validation](#field-validation) | [Changes from v1](#changes-from-the-first-version) |
| [Conference talk](#conference-talk) | | [What the results do not show](#what-the-results-do-not-show) | |

---

## Key results

| | Region 1<br/>Ten Thousand Islands | Region 2<br/>Everglades coast |
|---|:---:|:---:|
| **Ensemble** (physics-informed GNN + XGBoost), CV R² | 0.47 | 0.55 |
| **Ensemble, test R²** (declared second read) | **0.28** | **0.60** |
| XGBoost alone, test R² | 0.20 | 0.59 |
| 90% interval coverage, test | 91.5% | 89.1% |
| Median VM0033 deduction per node | **12.7%** | **2.1%** |
| Project coverage, VM0033 independent-error rule (95% CI) | 18% (2 to 52%) | 33% (20 to 49%) |
| Project coverage, empirical project calibration | 91% | 91% |

- **Registry rule:** even the upper end of the confidence interval of the VM0033 aggregation rule
  stays far below the 90% target, so the rule understates project uncertainty.
- **Market context:** the per-node deduction is of the same order as the **11.0%** median buffer
  of the seven blue carbon projects with issued credits (Berkeley VROD v2026-08). The two cover
  different risks, so this is context, not validation.
- **Calibration matters:** the label map's own uncalibrated intervals would withhold about **65%**.
- **Field cores:** the ensemble has the lowest site-level median error (**10.1%**), but paired
  bootstrap intervals of the differences include zero.

## Paper files

| File | Content |
|---|---|
| [`paper/main.pdf`](paper/main.pdf) | Revised paper, 5 pages, IEEE conference format |
| [`paper/main_changes.pdf`](paper/main_changes.pdf) | Changes from the first version (added text in blue, removed text in red) |
| [`paper/revision_preregistration.md`](paper/revision_preregistration.md) | Selection rules, deviations D1 to D9, addenda A to H |

## Conference talk

Presented at ICDABI 2026 (DATA'26), track *AI for Business & Industrial Data* (live talk, about
17 minutes). Besides the paper figures, the talk adds one figure built from the study's own
out-of-fold errors (spatial cross-validation on training and validation nodes; no test labels):

<p align="center">
  <img src="paper/figures/talk_correlated_errors.png" alt="Real project errors versus the independent-error level" width="820">
</p>

In one 2.5 km pseudo-project of R1 (259 nodes), the observed ensemble errors add up to a project
error of 5.5%; with the same error sizes but independent signs, as the VM0033 aggregation rule
assumes, they cancel to 0.4%. Across the 52 R1 pseudo-projects with at least 30 nodes, the actual
project error was a median **3.7 times** the independent-error level, and above it in 92% of
projects (`scripts/make_talk_fig_correlated_errors.py`, `results/talk_correlated_errors.json`).
This is an exploratory analysis that supports, and is not reported in, the paper.

## Tech stack

| Layer | Tools |
|---|---|
| **Language** | Python 3.12 |
| **Graph learning** | PyTorch 2.13, PyTorch Geometric 2.8 (GraphSAGE), physics-informed loss |
| **Tabular models** | XGBoost 3.4, scikit-learn 1.9 (random forest, linear, quantile models) |
| **Uncertainty** | Conformalized quantile regression, normalized conformal prediction, Mondrian calibration (`src/models/rev_uq.py`) |
| **Remote sensing and GIS** | rasterio, shapely, pyproj, pystac-client, Microsoft Planetary Computer, LERC |
| **Analysis and plots** | NumPy, pandas, SciPy, matplotlib |
| **Paper** | LaTeX (IEEEtran), BibTeX, latexdiff |
| **Reproducibility** | Fixed seeds, rules committed before each experiment, logged test reads, git history |

---

## Pipeline

```mermaid
flowchart LR
    subgraph IN["1 Inputs"]
        S2[Sentinel-2 L2A<br/>Feb + Aug 2026]
        DEM[USGS 3DEP lidar]
        TIDE[NOAA hourly<br/>water levels]
        SOC[Maxwell et al. 2023<br/>30 m SOC label]
        GMW[Global Mangrove Watch]
    end
    subgraph GR["2 Graph"]
        N[150 m nodes]
        E[8-neighbor edges<br/>split-induced subgraphs]
    end
    subgraph MO["3 Models"]
        GNN[Physics-informed<br/>GraphSAGE]
        XGB[XGBoost]
        ENS[Ensemble mean]
    end
    subgraph UQ["4 Uncertainty"]
        CQR[Conformal 90%<br/>intervals]
    end
    subgraph CR["5-6 Crediting"]
        VM[VM0033 Eq. 92<br/>deduction]
        BUF[Registry buffer<br/>unchanged]
        OUT[Credited t C]
    end
    S2 --> N
    DEM --> N
    TIDE --> N
    GMW --> N
    SOC --> N
    N --> E --> GNN --> ENS
    N --> XGB --> ENS
    ENS --> CQR --> VM --> OUT
    BUF --> OUT
    style IN fill:#EAF3FC,stroke:#3B6FB6,color:#1F3F8F
    style GR fill:#EAF3FC,stroke:#3B6FB6,color:#1F3F8F
    style MO fill:#EAF3FC,stroke:#3B6FB6,color:#1F3F8F
    style UQ fill:#FDEEF4,stroke:#E8749A,color:#7A2E4F
    style CR fill:#FDEEF4,stroke:#E8749A,color:#7A2E4F
```

**Crediting rule.** Estimation uncertainty and non-permanence risk stay two separate factors:

```
credited_i = stock_i x (1 - max(0, U_i - 0.20)) x (1 - b),   U_i = (upper_i - lower_i) / (2 stock_i)
```

**Physics prior.** The loss adds `lambda * MSE(y_z, P_z) + w * mean(ReLU(-y)^2)` with
`lambda = 0.1` (chosen by CV from 0.1, 0.3 and 1) and `w = 0.1`. The prior `P` is the mean of
min-max scaled tidal **inundation frequency** (share of 2025 hourly water levels above node
elevation) and red-edge canopy index (NDRE). It acts as a soft domain constraint: in CV it moved
R² from 0.39 to 0.41 (within one standard error), and its clear effect is stability across seeds
(SD 0.05 against 0.24 for the plain GNN).

---

## Study regions and data

| | Region 1 (R1) | Region 2 (R2) |
|---|---|---|
| Area | Ten Thousand Islands / Rookery Bay, SW Florida | Everglades NP coast (Lostmans to Shark River) |
| Bounding box | -81.90, 25.90, -81.40, 26.30 | -81.20, 25.30, -80.90, 25.60 |
| Nodes / directed edges | 10,124 / 64,366 | 22,326 / 143,034 |
| Train / val / test nodes | 6,465 / 1,583 / 2,076 | 14,499 / 3,351 / 4,476 |
| Tide gauge (NAVD88) | 8725114 Naples Bay North | 8723970 Vaca Key |
| Field soil cores (CCN, mangrove) | 56 | 31 |

Each node is a 5x5 block of the native 30 m SOC grid. Fine-grid cells (about 2.5 km) are
assigned whole to train, validation or test, and **no edge crosses a split**.

<a id="data-sources"></a>
<details>
<summary><b>Data sources</b></summary>

| Layer | Source |
|---|---|
| Sentinel-2 L2A | Copernicus, via Microsoft Planetary Computer |
| Elevation | USGS 3DEP seamless 1/3 arc-second; Copernicus GLO-30 (Planetary Computer) |
| Water levels | NOAA CO-OPS API, stations 8725114 and 8723970 |
| Mangrove extent | Global Mangrove Watch v3 (Bunting et al., Zenodo 6894273) |
| Soil organic carbon label | Maxwell et al. 2023, 30 m mangrove SOC 2020 (Zenodo 7729492) |
| Independent carbon map | Radabaugh et al. 2023, FWC SW Florida Carbon Stocks Map |
| Field soil cores | Smithsonian Coastal Carbon Network, Coastal Carbon Library v1.7.0 (2025) |
| Market data | Berkeley Carbon Trading Project, Voluntary Registry Offsets Database v2026-08 (CC BY 4.0) |
| Registry rule | Verra VM0033 v2.1 (2023), copy in `paper/refs_primary/` |

Provenance, access dates and licensing notes are in [`data/README.md`](data/README.md).
In the result files the column `sanderman` holds the Maxwell et al. (2023) label product.

</details>

---

## Evaluation protocol

```mermaid
flowchart TD
    A[Rules committed before each experiment] --> B[Spatial 5-fold CV on train+val<br/>features, models, physics prior]
    B --> C[Interval selection on a val-internal split]
    C --> D[First test read<br/>all pre-specified models]
    D --> F[Ensemble chosen on CV only]
    F --> G[Declared second test read<br/>one model, logged]
    D --> H[Exploratory checks<br/>field cores, project aggregation,<br/>market records, label-map intervals,<br/>confidence intervals]
    style A fill:#EAF3FC,stroke:#3B6FB6
    style B fill:#EAF3FC,stroke:#3B6FB6
    style C fill:#EAF3FC,stroke:#3B6FB6
    style D fill:#FBF0C2,stroke:#C9A227
    style G fill:#FBF0C2,stroke:#C9A227
    style F fill:#FDEEF4,stroke:#E8749A
    style H fill:#E3F4E8,stroke:#5BAA7A
```

- `scripts/run_revision_test_eval.py` is the only script that reads test labels. It writes
  `results/rev_test_RUN_STAMP.json` and refuses a second run without a logged reason.
- Every deviation (D1 to D9) and addendum (A to H) is logged in
  [`paper/revision_preregistration.md`](paper/revision_preregistration.md). Analyses run after
  the first test read are labelled exploratory.

---

## Accuracy

<p align="center">
  <img src="paper/figures/rev_fig1_results.png" alt="Results" width="900">
</p>

*(a) R² by model: R1 cross-validation (yellow, whiskers = fold range) and test in R1 (blue) and
R2 (pink), mean of 3 seeds. (b) Ensemble generalization. (c) 90% Winkler score on test,
dashed = trivial interval, labels = coverage.*

| Model | R1 CV | R1 test | R2 test |
|---|:---:|:---:|:---:|
| Constant (train mean) | n/a | -0.11 | -0.01 |
| Linear | 0.22 | -0.24 | 0.41 |
| Location only (RF on lon/lat) | 0.42 | -0.79 | 0.57 |
| Random forest | 0.38 | 0.03 | 0.58 |
| XGBoost | 0.41 | 0.20 | 0.59 |
| GraphSAGE | 0.39 | 0.07 (sd 0.24) | 0.58 |
| Physics-informed GraphSAGE | 0.41 | 0.17 (sd 0.05) | 0.56 |
| **Ensemble (physics-informed GNN + XGBoost)** | **0.47** | **0.28** | **0.60** |

The ensemble row comes from the declared second test read. Correcting a 9.18 km georeferencing
error in the first version's Sentinel-2 clip raised the GNN's CV R² from 0.17 to 0.26 on its own.

**Generalization.** One ensemble trained on both regions reaches CV R² 0.37 (R1) and 0.53 (R2).
Zero-shot transfer scores below zero, and recalibrating intervals on half of the target's
validation nodes restores 88% to 97% coverage. Training on both image dates raises the
cross-date rank agreement of predictions from 0.44 to 0.88.

## Intervals and crediting

| 90% intervals, test (CQR-GNN, global calibration) | Coverage | Mean width (t C/ha) | Winkler | ρ(width, abs error) |
|---|:---:|:---:|:---:|:---:|
| R1 CQR-GNN | 91.5% | 273 | 306 | 0.54 |
| R1 trivial interval | 89.3% | 274 | 372 | n/a |
| R2 CQR-GNN | 89.1% | 191 | 247 | 0.15 |
| R2 trivial interval | 87.8% | 282 | 369 | n/a |

The trivial interval is the training mean ± the conformal quantile of |y − mean| on the same
calibration nodes.

<p align="center">
  <img src="paper/figures/rev_fig2_deduction_map.png" alt="Per-node VM0033 deduction" width="560">
</p>

- Per-node VM0033 deduction: median **12.7%** (R1) and **2.1%** (R2); it concentrates in
  specific areas, which a flat buffer cannot represent.
- **Project level:** VM0033's independent-error aggregation covered 2 of 11 pseudo-projects in R1
  (18%, exact 95% interval 2 to 52%) and 15 of 45 in R2 (33%, 20 to 49%). A fully correlated bound
  covered 91% and 100%, and an empirical project-level calibration 91% in both regions. The
  reason is visible in the data: neighbouring errors share a sign (see [Conference talk](#conference-talk)).
- Interval width does not track mangrove loss between 2010 and 2020 (Spearman about -0.06), so
  the uncertainty deduction and the non-permanence buffer stay separate.
- As an illustrative standing-stock exposure, not credit issuance, the R1 deducted share equals
  0.71 Mt CO2e, or US$19.2 million at US$26.96 per credit.

## Field validation

| Source | Site-level median abs. error (25 sites, 87 cores) |
|---|:---:|
| Ensemble | **10.1%** |
| Physics-informed GNN | 11.6% |
| Maxwell et al. map | 14.1% |
| Constant regional mean | 15.0% |
| Radabaugh et al. map | 15.1% |
| XGBoost | 16.5% |

Paired bootstrap 95% intervals of the differences include zero (ensemble minus Maxwell map:
-4.0 points, -8.1 to 8.2), and the Maxwell map ranks sites best (Spearman 0.63 against 0.29).
The label map's own 95% limits contain all 13 deep cores but have a median half-width of 101% of
the estimate (`scripts/run_rev_g_label_pi.py`).

## What the results do not show

- The intervals are calibrated against held-out labels of a global map, not against measured
  carbon stocks; the label itself is a modeled product.
- The field advantage of the ensemble is within sampling uncertainty at 25 sites.
- Accuracy in R1 is moderate (test R² 0.28), and the ensemble comes from a declared second read.
- Pseudo-projects of 2.5 km are not registry projects; a registry-scale test is future work.
- Market transactions are not modeled; prices and buffers come from published records.

---

## Reproducing the results

Keep the virtual environment outside the repository: deeply nested Windows paths broke a DLL
load once.

**1. Environment**

```bash
python -m venv C:/venvs/bcpg
C:/venvs/bcpg/Scripts/pip install -r requirements.txt
C:/venvs/bcpg/Scripts/pip install xgboost openpyxl lerc pypdf
```

**2. Data** (the region 1 graph is in the repository; these rebuild features and references)

```bash
python src/data_pipeline/build_rev_features.py r1
python src/data_pipeline/build_hydroperiod.py r1
python src/data_pipeline/fetch_radabaugh_bgc.py r1
python src/data_pipeline/build_region2.py
python src/data_pipeline/build_rev_features.py r2
python src/data_pipeline/build_hydroperiod.py r2
python src/data_pipeline/fetch_radabaugh_bgc.py r2
python src/eval/nonpermanence.py r1
python src/eval/nonpermanence.py r2
```

**3. Model selection** (validation data only)

```bash
python scripts/run_rev_e1_cv.py
python scripts/run_rev_w2_physics_cv.py
python scripts/run_rev_e6_uq_val.py
python scripts/run_rev_f_cv.py
python scripts/run_rev_f2_uq.py
python scripts/run_rev_f34.py
```

**4. Test reads** (logged in `results/rev_test_RUN_STAMP.json`)

```bash
python scripts/run_revision_test_eval.py --dry-run
python scripts/run_revision_test_eval.py
python scripts/run_rev_f5_test2.py
```

**5. External checks, confidence intervals and figures**

```bash
python scripts/run_rev_d_field_validation.py
python scripts/run_rev_g_label_pi.py
python scripts/run_rev_h_intervals_on_results.py
python scripts/make_rev_fig0_framework.py
python scripts/make_rev_fig1.py
python scripts/make_rev_fig2_map.py
python scripts/make_talk_fig_correlated_errors.py
```

Seeds are fixed in every script (models 1 to 3, CV folds 20261007, region-2 split 20261009,
bootstrap 20261010). Sentinel-2, 3DEP and GLO-30 are read from Microsoft Planetary Computer
without an API key.

## Repository layout

```
blue-carbon-pinn-gnn/
├── data/
│   ├── raw/                     Sentinel-2 clips, GMW, SOC, tides, validation references,
│   │                            CCN cores, region2/, market/ (VROD)
│   └── processed/               graph_nodes*.csv, graph_edges*.npy (R1), region2/
├── src/
│   ├── data_pipeline/           region.py, build_graph.py, build_rev_features.py,
│   │                            build_hydroperiod.py, build_region2.py, fetch_*.py
│   ├── models/                  rev_common.py (data, graph regime, all model families),
│   │                            rev_uq.py (CQR, NCP, ensemble, conformal calibration)
│   ├── pricing/issuance.py      VM0033 Eq. 92 deduction, revenue sensitivity
│   └── eval/nonpermanence.py    GMW 2010-2020 mangrove loss per node
├── scripts/
│   ├── run_rev_e1_cv.py               model and feature selection (spatial CV)
│   ├── run_rev_w2_physics_cv.py       physics-prior selection
│   ├── run_rev_e6_uq_val.py           interval selection (validation)
│   ├── run_revision_test_eval.py      first test read (both regions, transfer)
│   ├── run_rev_f_cv.py                ensemble selection (spatial CV)
│   ├── run_rev_f2_uq.py, run_rev_f34.py  interval method, field sites, pooled model
│   ├── run_rev_f5_test2.py            declared second test read (ensemble only)
│   ├── run_rev_b_*, run_rev_c*_*      local calibration, fix-only ablation, two-date training
│   ├── run_rev_d_field_validation.py  CCN field-core comparison
│   ├── run_rev_e_*                    issuance under every interval method, project aggregation
│   ├── run_rev_f_market_vrod.py       market-practice check
│   ├── run_rev_g_label_pi.py          label-map intervals against field cores
│   ├── run_rev_h_intervals_on_results.py  confidence intervals for reported numbers
│   ├── make_talk_fig_correlated_errors.py  talk figure from real out-of-fold errors
│   └── make_rev_fig*.py               figures
├── paper/                       main.tex, main.pdf, main_changes.pdf, references.bib,
│                                figures/, revision_preregistration.md
└── results/                     every CSV/JSON behind a number in the paper
```

---

## Changes from the first version

| Issue in the first version | What changed |
|---|---|
| Negative test R² | Found and fixed a 9.18 km georeferencing error in the Sentinel-2 clip (reproduced from clean inputs); richer features; ensemble test R² 0.28 (R1) and 0.60 (R2) |
| No RF / XGBoost | Added RF, XGBoost, RF with spatial lag, a location-only reference and a constant baseline |
| Split leakage | Training and inference on split-induced subgraphs; 2-hop interior subset reported |
| Physics prior built on NDVI and gauge distance | Replaced by a hydroperiod prior (3DEP lidar and a year of NOAA water levels) plus red-edge |
| Circular prior-generalization metric | Dropped as evidence; independent checks against 87 field soil cores and a field-calibrated map |
| CQR did not use the physics model | Physics-informed quantile GNN added; crediting reported under every interval method |
| Intervals wider than the label SD | Compared against a trivial conformal interval; adaptivity and Winkler score reported |
| Arbitrary pricing parameters, risks conflated | VM0033 Eq. 92 deduction, kept separate from the registry buffer; tested against mangrove loss and market buffers |
| Single region, single date | Second region, second Sentinel-2 date, two-date training, pooled two-region model |
| Mondrian dropped after seeing test | Both calibrations reported throughout |
| Contribution unclear | Research-gap table and pipeline figure; contributions stated in the introduction |
| Claims stronger than the evidence | Exact confidence intervals for project coverage, paired bootstrap for field errors, label-calibrated wording, every reference checked against its source |

---

<div align="center">

**Nahla Nabil** · Faculty of Computer Studies, Arab Open University, Kingdom of Bahrain

</div>

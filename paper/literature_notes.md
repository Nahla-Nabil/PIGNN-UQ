# Literature & Baseline Scan

Compiled 2026-08-29. Scope: enough to (a) place the paper against the closest
prior art and state a clear delta, (b) name a validation reference for the
pilot region, (c) name the business-side baseline we are trying to improve
on, and (d) justify the UQ method choice in [`uq_decision.md`](uq_decision.md).
Not a full systematic review — this is a 5-page conference paper.

---

## 1. PINN + GNN hybrid architectures (general SciML, not necessarily environmental)

- **Raissi, Perdikaris & Karniadakis (2019)**, "Physics-informed neural
  networks: A deep learning framework for solving forward and inverse
  problems involving nonlinear partial differential equations," *Journal of
  Computational Physics* 378:686–707. The foundational PINN paper: encode a
  PDE residual as an additional loss term. Our physics term (mass-balance)
  is a direct descendant of this idea, just moved from a PDE residual to a
  discrete conservation-law residual on a graph.
  [doi:10.1016/j.jcp.2018.10.045](https://doi.org/10.1016/j.jcp.2018.10.045) ·
  [code](https://github.com/maziarraissi/PINNs)

- **Sanchez-Gonzalez, Godwin, Pfaff, Ying, Leskovec & Battaglia (2020)**,
  "Learning to Simulate Complex Physics with Graph Networks," *ICML 2020*,
  PMLR 119:8459–8468. Graph Network-based Simulator (GNS): particles as
  nodes, learned message passing as the dynamics. Physics is *encoded* in
  the architecture (locality, permutation invariance) rather than added as
  an explicit loss term — this is the key architectural precedent for
  representing a physical system as a graph, but it is not "physics
  *informed*" in the PINN loss-term sense.
  [arXiv:2002.09405](https://arxiv.org/abs/2002.09405)

- **"Physics-Informed Graph Neural Network for Spatial-temporal Production
  Forecasting"** (2022). Closest structural analog we found outside
  environmental science: a spatiotemporal graph + physics-informed loss
  predicting a resource quantity (oil production) over a well network. Same
  shape as our problem (replace wells/production with mangrove
  patches/carbon).
  [arXiv:2209.11885](https://arxiv.org/pdf/2209.11885)

- **"A physics-informed graph neural network conserving linear and angular
  momentum for dynamical systems"** (*Nature Communications*, 2025) —
  "DynamicalGraphNet." Enforces pairwise conservation laws between nodes
  using edge-local reference frames. Useful precedent for *how* to encode a
  conservation constraint on graph edges rather than only as a global scalar
  penalty — relevant to how we should formulate the accretion/export terms
  between adjacent mangrove-patch nodes.
  [nature.com/articles/s41467-025-67802-5](https://www.nature.com/articles/s41467-025-67802-5)

- **"Scalable algorithms for physics-informed neural and graph networks"**
  (survey), *Cambridge / arXiv:2205.08332*. Reviews physics-informed graph
  networks built via graph exterior calculus (discrete differential
  operators on graphs) — a more principled route to a discretized
  mass-balance operator than an ad hoc penalty term; worth citing as the
  theoretical basis if we have room to justify our operator formally.
  [arXiv:2205.08332](https://arxiv.org/pdf/2205.08332)

- **Environmental PINN-GNN precedents (closest actual prior art):**
  physics-informed GNNs for **flood forecasting** embed mass-conservation
  directly into the loss of a spatiotemporal GNN over a river/floodplain
  graph — structurally almost identical to what we propose, just a
  different conserved quantity and a different graph (river network /
  floodplain cells vs. mangrove patches):
  - Taghizadeh et al. (2025), "Interpretable physics-informed graph neural
    networks for flood forecasting," *Computer-Aided Civil and
    Infrastructure Engineering* (Wiley).
    [onlinelibrary.wiley.com/doi/full/10.1111/mice.13484](https://onlinelibrary.wiley.com/doi/full/10.1111/mice.13484)
  - "DUALFloodGNN: Physics-informed Graph Neural Network for Operational
    Flood Modeling" — mass conservation embedded in the GNN loss for flood
    depth prediction. [arXiv:2512.23964](https://arxiv.org/pdf/2512.23964)
  - Bindas et al. / PNNL–HESS (2022), "A graph neural network (GNN)
    approach to basin-scale river network learning: the role of
    physics-based connectivity and data fusion," *HESS* 26:5163.
    [hess.copernicus.org/articles/26/5163/2022](https://hess.copernicus.org/articles/26/5163/2022/)
  - "Physics-Guided Recurrent Graph Networks for Predicting Flow and
    Temperature in River Networks" (2020).
    [arXiv:2009.12575](https://arxiv.org/pdf/2009.12575)

**Our delta:** every PINN-GNN paper we found that uses a genuine
conservation-law loss term operates on hydrology (flood/streamflow)
networks. None target blue-carbon / mangrove biomass-soil carbon dynamics,
and none couple the physics-informed prediction to a downstream **priced,
risk-adjusted financial instrument** via UQ. That combination — mass-balance
GNN + calibrated UQ + dynamic credit pricing — is, as far as this scan
found, novel.

---

## 2. Mangrove / blue carbon biomass estimation from Sentinel-2 or similar remote sensing

- **Simard, Fatoyinbo, Smetanka et al. (2019)**, "Global Mangrove
  Distribution, Aboveground Biomass, and Canopy Height," ORNL DAAC,
  doi:10.3334/ORNLDAAC/1665. 30 m global AGB/canopy-height/basal-height maps
  from SRTM + GLAS + region-specific allometry, per-country GeoTIFFs
  (includes USA/Florida). **This is the standard global reference product
  we would validate against** if not for the access gap noted in
  [`data/README.md`](../data/README.md) (NASA Earthdata login required).
  [earthdata.nasa.gov/data/catalog/ornl-cloud-cms-global-map-mangrove-canopy-1665-1.3](https://www.earthdata.nasa.gov/data/catalog/ornl-cloud-cms-global-map-mangrove-canopy-1665-1.3)

- **Radabaugh, Moyer, Chappel et al. (2023)**, "A Spatial Model Comparing
  Above- and Belowground Blue Carbon Stocks in Southwest Florida Mangroves
  and Salt Marshes," *Estuaries and Coasts* 46:1536–1556,
  doi:10.1007/s12237-023-01217-7. 30 m spatial carbon-stock model built from
  field plots along 360 km of Southwest Florida coast, **Tampa Bay to the
  Everglades — directly covers our pilot AOI** (Ten Thousand
  Islands/Rookery Bay). Reports mangrove AGB+BGB ≈ 393.9 ± 107.1 Mg C ha⁻¹.
  This is our primary **field-validated regional reference** for sanity-
  checking predicted carbon stocks.
  [doi.org/10.1007/s12237-023-01217-7](https://doi.org/10.1007/s12237-023-01217-7)

- **Rovai, Twilley et al.**, "Global controls on carbon storage in mangrove
  soils" — coastal-ecosystem-setting model showing mangrove soil organic
  carbon (SOC) stocks vary systematically by geomorphic setting and are
  mis-estimated by up to 50–86% by uniform global averages. Motivates using
  a spatially explicit SOC layer (below) rather than a constant per-hectare
  assumption in our mass-balance term.
  [digitalcommons.fiu.edu/fce_lter_journal_articles/513](https://digitalcommons.fiu.edu/fce_lter_journal_articles/513/)

- **Sanderman/Worthington/WHRC et al. (2023)**, "Global mangrove soil
  organic carbon stocks dataset at 30 m resolution for the year 2020 based
  on spatiotemporal predictive machine learning," *PMC10562832* /
  ScienceDirect. Ensemble ML (10,331 field soil samples) predicting SOC
  content + bulk density to 1 m depth. **We pulled a real windowed sample of
  the companion typology layer from this dataset for our AOI** — see
  [`data/README.md`](../data/README.md).
  [ncbi.nlm.nih.gov/pmc/articles/PMC10562832](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10562832/)

- **Lagomasino, Fatoyinbo et al.**, "Estimating Mangrove Canopy Height and
  Above-Ground Biomass in the Everglades National Park with Airborne LiDAR
  and TanDEM-X Data." Region-specific (Everglades) AGB/height reference
  (~250 Mg·ha⁻¹ max AGB, ~25 m max canopy height) — a second, independent
  cross-check for our AOI if Simard/ORNL access remains blocked.
  [researchgate.net/publication/318293064](https://www.researchgate.net/publication/318293064_Estimating_Mangrove_Canopy_Height_and_Above-Ground_Biomass_in_the_Everglades_National_Park_with_Airborne_LiDAR_and_TanDEM-X_Data)

- **Sentinel-2 + ML methodology precedents** (establish that RF/XGBoost on
  spectral indices is the current standard we should beat or at least
  match): mangrove AGB in Zhejiang (multi-algorithm ML), Hainan
  (multi-source RS + ML), North Vietnam SOC (Sentinel-2 + ML,
  doi:10.1080/15481603.2020.1857623), Bali (Sentinel-2 vegetation indices),
  and Matang Reserve, Malaysia (Sentinel-2 + ALOS-2, R²=0.735,
  RMSE=46.8 t/ha) — consistent picture of per-pixel tabular regression as
  the field's dominant approach.
  [doi.org/10.3390/f16101517](https://doi.org/10.3390/f16101517) ·
  [tandfonline.com/doi/full/10.1080/15481603.2020.1857623](https://www.tandfonline.com/doi/full/10.1080/15481603.2020.1857623)

**Our delta:** the field's dominant approach is per-pixel tabular ML
(RF/XGBoost/SVM) with no explicit spatial structure and no physical
constraint; the graph representation plus mass-balance loss is the gap we
fill, validated against Radabaugh et al. (2023) as regional ground truth.

---

## 3. Carbon credit buffer-pool / non-permanence risk discounting methodologies (business-side baseline)

- **Verra VM0033**, "Methodology for Tidal Wetland and Seagrass
  Restoration," v2.1. Governs blue carbon credit issuance for tidal wetland
  restoration; requires the VCS AFOLU Non-Permanence Risk Tool specifically
  to assess **erosion and submergence risk from sea-level rise** — i.e.
  hydrological risk is already explicitly in scope for this methodology
  family, which supports tying our tidal/mass-balance term to the pricing
  layer.
  [verra.org/methodologies/vm0033-methodology-for-tidal-wetland-and-seagrass-restoration-v2-1](https://verra.org/methodologies/vm0033-methodology-for-tidal-wetland-and-seagrass-restoration-v2-1/)

- **VCS AFOLU Non-Permanence Risk Tool** (used by both VM0033 and VM0007).
  The actual scoring rubric: a checklist-based risk score (project
  management, financial, legal, social, natural risks) maps to a fixed
  buffer-pool contribution percentage, reassessed periodically (not
  continuously). This is the concrete mechanism we are proposing to
  **augment with a continuous, model-driven signal**.
  [verra.org/wp-content/uploads/2018/03/Tool-for-AFOLU-Non-Permanence-Risk-Analysis-and-Buffer-Determination.pdf](https://verra.org/wp-content/uploads/2018/03/Tool-for-AFOLU-Non-Permanence-Risk-Analysis-and-Buffer-Determination.pdf)

- **Plan Vivo Non-Permanence Buffer (v4.0)**. Flat 15–20% buffer withheld
  into a shared pool; risk reassessed only every ~10 years, with annual
  monitoring in between. Same structural pattern as VCS: static,
  infrequently updated, not resolved at sub-project spatial granularity.
  [planvivo.org/buffer](https://www.planvivo.org/buffer)

- **Plan Vivo FAQ**, "How are risks to permanence minimised?" — plain-
  language description of the pooled-buffer mechanism, useful for the
  paper's related-work framing of "how the industry currently prices
  non-permanence risk."
  [planvivo.org/faqs/how-are-risks-to-permanence-minimised](https://www.planvivo.org/faqs/how-are-risks-to-permanence-minimised)

- **Critical evidence that current buffer pools are miscalibrated:**
  "Current forest carbon offset buffer pools do not adequately insure
  against disturbance-driven carbon losses" (bioRxiv, 2024). Directly
  motivates our claim: today's buffer percentages are coarse, rule-based,
  and not empirically tied to observed reversal risk at the
  project/sub-project level — the gap a calibrated, per-region UQ signal is
  positioned to close.
  [biorxiv.org/content/10.1101/2024.03.28.587000](https://www.biorxiv.org/content/10.1101/2024.03.28.587000.full.pdf)

**Our delta / baseline comparator for the paper:** VM0033 / Plan Vivo buffer
percentages are our business-side baseline. The paper's pricing section
should show, even qualitatively on the pilot region, how a per-node
UQ-derived discount would differ from the flat 15–20% buffer these
standards currently apply uniformly.

---

## 4. Conformal prediction / deep ensembles for environmental & geospatial UQ

(Preempting the known weakness of MC Dropout — see
[`uq_decision.md`](uq_decision.md) for the full comparison and recommendation.)

- **Gal & Ghahramani (2016)**, "Dropout as a Bayesian Approximation:
  Representing Model Uncertainty in Deep Learning," *ICML 2016*. Origin of
  MC Dropout. Cite as the method we are evaluating against and explicitly
  note its known failure mode: dropout-derived uncertainty approximates a
  specific variational posterior only under assumptions that rarely hold in
  practice, and it is known to be poorly calibrated, especially under
  covariate/distribution shift — exactly the regime a growing/degrading
  mangrove patch under sea-level rise represents.
  [arXiv:1506.02142](https://arxiv.org/abs/1506.02142)

- **Lakshminarayanan, Pritzel & Blundell (2017)**, "Simple and Scalable
  Predictive Uncertainty Estimation using Deep Ensembles," *NeurIPS 2017*.
  Origin of deep ensembles; empirically the "gold standard" for calibrated
  predictive distributions among the non-conformal methods, at the cost of
  training M full models.
  [arXiv:1612.01474](https://arxiv.org/pdf/1612.01474)

- **Romano, Patterson & Candès (2019)**, "Conformalized Quantile
  Regression," *NeurIPS 2019*. CQR: wraps any quantile regressor with
  split-conformal calibration for **finite-sample, distribution-free
  coverage**, and is adaptive to heteroscedasticity (unlike vanilla split
  conformal's constant-width intervals) — directly relevant since
  uncertainty should differ between a stable interior mangrove patch and an
  eroding fringe patch. This is the method we recommend as primary; see
  `uq_decision.md`.
  [arXiv:1905.03222](https://arxiv.org/abs/1905.03222) ·
  [code](https://github.com/yromano/cqr)

- **"Uncertainty quantification for probabilistic machine learning in earth
  observation using conformal prediction"** (*Scientific Reports*, 2024).
  Earth-observation-specific validation that conformal prediction gives
  valid, model-agnostic coverage for EO regression/classification tasks at
  local-to-global extents — the closest direct precedent for applying CP to
  a remote-sensing-driven GNN.
  [nature.com/articles/s41598-024-65954-w](https://www.nature.com/articles/s41598-024-65954-w)

- **"Uncertainty quantification for forest attribute maps with conformal
  prediction and k-nearest neighbor method"** (*Remote Sensing of
  Environment*, 2025). CP applied to Sentinel-2 + airborne laser scanning
  forest attribute (volume, height) mapping — near-identical data modality
  to ours (Sentinel-2-driven structural/biomass regression), useful as a
  methods template.
  [sciencedirect.com/science/article/pii/S0034425725001622](https://www.sciencedirect.com/science/article/pii/S0034425725001622)

- **"Spatial-Aware Conformal Prediction for Trustworthy Hyperspectral Image
  Classification."** Important caveat: standard split conformal prediction
  assumes exchangeability, which spatially autocorrelated data (our graph
  nodes) violates; spatial-aware / clustered calibration variants address
  this. **We must use a spatially-blocked calibration split**, not an i.i.d.
  random split, or our coverage guarantee is not actually valid — flag this
  explicitly as a methodological requirement, not an optional refinement.
  [researchgate.net/publication/390691348](https://www.researchgate.net/publication/390691348_Spatial-Aware_Conformal_Prediction_for_Trustworthy_Hyperspectral_Image_Classification)

- **"Quantifying Uncertainty in Spatial Prediction for Nonstationary
  Spatial Processes"** (2026). Reinforces the same point from the
  spatial-statistics side: coverage validity under nonstationarity needs
  explicit handling, relevant given our AOI spans open water, dense
  interior mangrove, and eroding fringe in one graph.
  [tandfonline.com/doi/full/10.1080/24694452.2026.2639721](https://www.tandfonline.com/doi/full/10.1080/24694452.2026.2639721)

---

## Summary table: where each section feeds the paper

| Section | Feeds into | 5-page paper use |
|---|---|---|
| 1. PINN-GNN prior art | Related Work | State delta vs. flood-forecasting PINN-GNNs (closest analog) |
| 2. Mangrove biomass RS | Related Work + Results validation | Radabaugh et al. (2023) as the number(s) we sanity-check predictions against |
| 3. Buffer-pool methodologies | Introduction (motivation) + Pricing section | VM0033/Plan Vivo flat buffer % as the baseline our dynamic pricing is compared to |
| 4. Conformal/ensemble UQ | Methods (UQ module) | Justifies CQR choice; flags spatial-exchangeability requirement as a methods constraint |

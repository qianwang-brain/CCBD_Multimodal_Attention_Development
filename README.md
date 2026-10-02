# Multimodal developmental signatures of attention across childhood and adolescence extend to ADHD

Selected analysis scripts accompanying the manuscript with the same title. The repository contains selective-attention (HX) and sustained-attention (two-task) post-fusion analyses, environmental item processing/PCA, and independent PKU ADHD prediction and longitudinal analyses.

The discovery folders contain only the scripts selected by the authors on 2 October 2026. Their analysis types can be mapped to manuscript sections, but exact historical figure/table provenance and numerical reproduction have not been verified. Author-authorized corrections were applied on 2 October 2026; clinical-validation issues remain unresolved. See [CODE_REVIEW.md](CODE_REVIEW.md) before using the outputs.

## Multimodal fusion

Multimodal fusion was performed in the [Fusion ICA Toolbox (FIT)](https://trendscenter.org/software/fit/). Reference-guided multimodal canonical correlation analysis (MCCAR) jointly integrated resting-state EEG relative power, grey-matter volume (GMV) and resting-state functional connectivity strength (FCS). Separate fusion models used selective- and sustained-attention performance as behavioral references to identify attention-linked modality-specific components and participant-level loadings.

The curated discovery scripts in this repository analyse the resulting fusion outputs. Obtain FIT separately to perform the upstream fusion analysis; the selected folders do not include the fusion fitting implementation. Exact toolbox version and study-specific fusion settings should be documented alongside any reproduced run.

## Study datasets

The following descriptions and sample counts refer to the manuscript's analysed samples after quality control, rather than the full size of either study.

**CCBD discovery cohort.** The Chinese Child Brain Development (CCBD) project is a population-based, multicentre study of brain and cognitive development in Chinese school-age children and adolescents. It uses a nationwide stratified sampling framework across geographically and socioeconomically diverse regions, with harmonized assessments of brain structure and function, cognition, behavior and environmental exposures. The discovery sample comprised 3,736 participants aged 6.04–18.99 years with EEG and MRI data. Selective-attention data were available for 1,965 participants and sustained-attention data for 1,882; 1,856 completed both tasks. CCBD was used to identify attention-linked multimodal signatures and examine their developmental variation, environmental associations and MINI ADHD screening-group differences.

**PKU clinical validation cohort.** The independent single-centre cohort was recruited at Peking University Sixth Hospital and comprised 529 participants aged 6.6–17.8 years after quality control. Discovery-derived attention features were used to predict dimensional clinical symptoms and executive-function measures assessed with ADHD-RS, Conners and BRIEF. An exploratory longitudinal subset included 43 participants with ADHD assessed at baseline and after 8 weeks of methylphenidate treatment, for analyses relating brain-feature changes to symptom improvement.

The manuscript reports ethics approval from Beijing Normal University and Peking University Sixth Hospital, parental/legal-guardian informed consent and age-appropriate participant assent or consent. Participant-level data are not included in this repository; access must be arranged with the respective study data custodians. See [Data availability](#data-availability) and [data/README.md](data/README.md).

## Repository structure

```text
code/
├── code_github_hx/              # Eight selected selective-attention scripts
├── code_github_twotask/         # Seven selected sustained-attention scripts
├── github_code_env/            # Environmental item audit, domain PCA and loadings
└── adhd_pku/
    ├── prediction/             # Clinical prediction, permutation and scatter plots
    └── treatment/              # Follow-up partial correlations and plots
data/README.md                  # Input schema and access notes; no data included
docs/FILE_INVENTORY.md           # All 26 distributed source files and I/O statements
docs/RUNNING.md                  # Working-directory and dependency requirements
docs/RELEASE_CHANGES.md          # Curation and preservation record
```

The former `新建文件夹` subdirectories have been archived outside this upload folder. Full-sample/group fusion fitting, the MCCAR solver, repeated CV, paired bootstrap and discovery EEG/MNI renderers are not distributed in these two curated folders. This is a selected analysis-code release, not a complete end-to-end reproduction pipeline.

## Selected discovery scripts and manuscript correspondence

| Script | HX | Sustained | Function and candidate correspondence |
|---|---|---|---|
| `1_results_adjust.m` | Yes | Yes | Orient previously fitted components and derive thresholded feature indices; upstream to map/feature analyses |
| `2_correlations_plot.m` | Yes | Yes | Loading–attention associations with sex/TIV adjustment and age-adjusted sensitivity; Fig. 2d/h analysis type |
| `2_1_correlations_age.m` | Yes | — | Loading/raw-feature age plots and participant-level GAM export; supplementary age analysis candidate |
| `2_interactions_plot_age.m` | Yes | — | Loading × continuous age models; Fig. 3a / Supplementary Table 4 candidate |
| `2_interactions_plot.m` | — | Yes | Loading × continuous age models; Fig. 4a / Supplementary Table 8 candidate |
| `2_interactions_plot_ANCOVA.m` | Yes | Yes | Stage-specific extreme-decile groups, ANCOVA/Tukey–Kramer and within-stage Welch tests; Fig. 3b/4b and Supplementary Tables 5–7/9–11 analysis types |
| `2_interactions_env.m` | Yes | Yes | Age × environment models, standardized coefficients and BH correction; Fig. 5a/c statistical inputs |
| `2_interactions_plot_env.m` | Yes | Yes | Curves for interactions selected by the preceding script; Fig. 5b/d curve candidates |
| `4_results_group_difference_mini.m` | Yes | Yes | MINI ADHD screen-positive versus all screen-negative loading comparison described in Methods |

These correspondences are inferred from code and manuscript definitions, not from a verified match to the final numeric tables or rendered panels. Environmental PC1 loading tables come from `github_code_env/4_check_loadings.m`; the final coefficient and loading-bar figure renderers are not identified among the selected scripts.

## Requirements

- MATLAB; Statistics and Machine Learning Toolbox (`glmfit`, `fitlm`, `anovan`, `multcompare`, `ttest2`, `ksdensity`, `pca`); Bioinformatics Toolbox (`mafdr`). Selected plots call the original `gretna_plot_regression_wq` helper, which is not distributed in the curated folders and must be supplied separately.
- Python 3.10 or later for the syntax used by PKU scripts; NumPy, pandas, SciPy, scikit-learn, LightGBM, matplotlib and openpyxl. XGBoost is optional when requested via the prediction CLI.
- R with base plotting and Cairo PDF support for the alternative follow-up heatmap script. Fonts specified by the original scripts must be available.

No original software version record was supplied. Seven distributed Python files passed AST parsing under Python 3.12. MATLAB `checkcode` could not be started successfully, R compilation was not performed, and numerical analyses were not executed.

## Data availability

CCBD and PKU clinical data are not provided. Obtain authorized access from the respective study data custodians under their access/ethics policies. Input MAT files, participant IDs, clinical scores and individual-level outputs must remain outside public Git history. [data/README.md](data/README.md) describes the expected schemas.

## Running the analyses

Configure each script's private input/output paths and working directory. `/path/to/project/` and `/path/to/pku_prediction` are placeholders. Keep HX and sustained output directories separate because filenames are shared. Create output folders where the scripts do not do so.

Discovery scripts begin from externally generated fusion outputs (`1_30_a/s/r/indmat.mat`) and authorized modality/covariate data. Run `1_results_adjust.m` only after confirming its component-selection assumptions and thresholds, then choose the required correlation, interaction, ANCOVA or MINI branch. `2_interactions_env.m` writes `results_interaction_age_env2.mat`, consumed by `2_interactions_plot_env.m`; both scripts now run the six-PC1 analysis only.

Environmental preparation follows item collection → author-reviewed `item_direction_audit_manual_check.csv` → recoding builder → domain PCA → loading-table export. The manual audit is required and is not generated by the item collection script.

From `code/adhd_pku/prediction/`, with private prediction MAT files available:

```bash
python analysis_svr_three_models.py --models SVR LightGBM --outdir analysis_outputs_two_model_multi_family_allsubjects_svr_lgbm_tuned_dqia
python random_score_permutation_100x_lightgbm_sigtargets.py
```

The preserved default is 100 permutations; the manuscript states 1,000. Confirm the historical run before using these commands as a reproduction protocol. Configure the scatter script's `BASE` to this private working directory. Existing `--n-repeats` changes the permutation output folder for non-default repeat counts, so downstream paths must be set consistently.

Follow-up scripts use their own directory for private paired MAT inputs. Run `analyze_followup_med1_partialcorr.py`, then the desired Python renderer. Run the R heatmap script from the treatment directory because it uses `getwd()`. Review the partial-correlation p-value issue before trusting the generated FDR hits.

## Inputs and outputs

| Module | Main inputs | Main outputs |
|---|---|---|
| Discovery post-fusion | Fusion loadings/maps and selected indices; `data2.mat`; modality/ROI/channel labels | Oriented component files, feature indices, regression/group statistics and plots |
| Developmental-stage analysis | Loadings, attention, sex, TIV and age-group `IDs` | ANCOVA/Tukey–Kramer and Welch statistics, box/scatter figures |
| Environment | Authorized environmental data and manual audit; `PC1_table`; loadings/covariates | Oriented items, domain PCA/loading tables, interaction MAT and curves |
| PKU prediction | `feature`, `scores`, `data_name`, `label`, `inf` in HX/Tt MAT files | Raw/residual prediction summaries, individual OOF predictions and permutation results |
| PKU treatment | Paired clinical/brain features, covariates and medication indicator | Partial-correlation/FDR CSV/XLSX, heatmaps and scatter PDFs |

Detailed per-file I/O statements appear in [FILE_INVENTORY.md](docs/FILE_INVENTORY.md). Individual predictions and GAM exports are private outputs, not public data assets.

## Reproducibility and manuscript consistency

- EEG, GMV and FCS feature selection all use `abs(feature_z) > 2` in both discovery folders. GMV/FCS values with |Z|<=2 are zeroed before selecting indices.
- Within-stage Welch comparisons now apply BH-FDR across 12 tests per attentional phenotype. Figure stars use q<0.05; `within_stage_Welch_FDR.csv` and `developmental_stage_statistics.mat` retain raw p and adjusted q values.
- Environmental GMV statistical and curve models now include centered TIV. Participant IDs are matched on both sides, with duplicate-ID and alignment assertions.
- Loading × age models now include TIV for GMV within the single three-modality loop, save the corrected statistics once, and export PDF/PNG figures.
- Sustained environmental analysis now contains only the six-domain PC1 branch; the extra raw-item branch was removed.

These changes implement the author-authorized corrections; they may change feature indices, estimates and significance. Re-run with authorized data and reconcile the new outputs with final manuscript values. Original workspace sources and restricted data remain untouched. Exact historical paper values are not certified.

Known 12-test BH values, FDR ordering, interaction coefficient positions, prediction dimensions and two-sided ID matching passed synthetic Python checks. These check equivalent numerical operations, not MATLAB execution. MATLAB failed to start; no real-data rerun was performed.

Clinical prediction tuning provenance, final FDR, permutation count/tail and follow-up partial-correlation p-values remain unresolved. Component selection/sign assumptions and final environmental item counts also need provenance verification. See [CODE_REVIEW.md](CODE_REVIEW.md).

## Citation

*Multimodal developmental signatures of attention across childhood and adolescence extend to ADHD*, Qian Wang et al. Citation information will be updated upon publication; no publication DOI was supplied.

## Contact and license

Use GitHub issues for code questions and the associated manuscript for scientific correspondence. No license was supplied; authors must select one and confirm redistribution permissions before public release.

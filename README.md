# EEG-fMRI-sMRI signatures Links Attention Development To Clinical Variation in ADHD

Analysis code accompanying the manuscript. The study combines EEG, functional MRI and structural MRI to examine selective and sustained attention development and their relevance to clinical variation in ADHD.

## Data

- **CCBD:** The Chinese Child Brain Development project, a multicentre developmental cohort. This study includes 3,736 participants with multimodal brain data.
- **PKU6:** An independent clinical cohort from Peking University Sixth Hospital, including 529 participants and a 43-participant methylphenidate follow-up subset.

Participant-level data are not included. Access should be requested from the respective study data custodians.

## Multimodal fusion

Fusion analyses were performed using the [Fusion ICA Toolbox (FIT)](https://trendscenter.org/software/fit/). The discovery scripts use the resulting attention-guided multimodal components and loadings.

## Code structure

```text
code/
├── code_github_hx/       # Selective-attention analyses
├── code_github_twotask/  # Sustained-attention analyses
├── github_code_env/     # Environmental measures and PCA
└── adhd_pku/
    ├── prediction/      # Clinical prediction and permutation testing
    └── treatment/       # Treatment-change analyses
```

## Usage

Discovery and environmental analyses use MATLAB. PKU analyses use Python with NumPy, pandas, SciPy, scikit-learn, LightGBM and openpyxl. Configure private data paths before running the scripts.

From the prediction folder:

```bash
python predict_clinical_scores.py
python permutation_test_predictions.py
```

From the treatment folder:

```bash
python analyze_treatment_changes.py --data-dir /path/to/private/followup
```

See [the running guide](docs/RUNNING.md) for setup and [PKU analysis notes](docs/PKU_ANALYSES.md) for the current analysis workflow.

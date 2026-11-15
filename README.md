# Multimodal brain signatures of selective and sustained attention across development and their clinical relevance to ADHD

Analysis code accompanying the manuscript. The study integrates EEG, structural MRI and resting-state fMRI to investigate selective and sustained attention across development and their clinical relevance to ADHD.

## Data

Four independent cohorts were included:

- **CCBD:** Chinese Children Brain Development cohort; 3,736 children aged 6–18 years with multimodal EEG, structural MRI and resting-state fMRI data. Used for discovery and developmental analyses.
- **ABCD:** Adolescent Brain Cognitive Development Study; an independent US cohort used for external validation of GMV and FCS signatures.
- **HBN:** Healthy Brain Network; an independent US cohort used for external validation of EEG signatures.
- **PKU6:** Peking University Sixth Hospital clinical cohort (n = 529), including a 43-participant, 8-week methylphenidate follow-up subset. Used for clinical prediction and longitudinal analyses.

Participant-level data are not included. Access is subject to the respective data-use and ethical requirements.

## Multimodal fusion

Fusion analyses were performed using the [Fusion ICA Toolbox (FIT)](https://trendscenter.org/software/fit/). Discovery scripts use the resulting attention-guided multimodal components and loadings.

## Code structure

```text
code/
├── code_github_hx/       # Selective-attention analyses
├── code_github_twotask/  # Sustained-attention analyses
├── github_code_env/      # Environmental measures and PCA
└── adhd_pku/
    ├── prediction/       # Clinical prediction and permutation testing
    └── treatment/        # Treatment-change analyses
```

## Usage

Discovery and environmental analyses use MATLAB. PKU6 analyses use Python with NumPy, pandas, SciPy, scikit-learn, LightGBM and openpyxl. Configure data paths before running.

```bash
python predict_clinical_scores.py
python permutation_test_predictions.py
```

See [docs/RUNNING.md](docs/RUNNING.md) for setup and [docs/PKU_ANALYSES.md](docs/PKU_ANALYSES.md) for analysis details.

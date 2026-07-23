# EEG Biomarkers using MSC and wPLI

## Overview

Electroencephalography (EEG) is a non-invasive neuroimaging technique that records the brain's electrical activity with high temporal resolution, making it a valuable tool for studying functional brain connectivity. Alterations in connectivity patterns have been associated with several neurodevelopmental disorders, including Intellectual Disability (IDD), highlighting the need for reliable EEG-based biomarkers.

This project investigates functional connectivity differences between Individuals with Intellectual Disabilities (IDD) and Typically Developing Controls (TDC) using EEG recordings. The analysis employs two widely used connectivity metrics: **Magnitude Squared Coherence (MSC)** to quantify frequency-domain synchronization and **Weighted Phase Lag Index (wPLI)** to measure phase synchronization while reducing the influence of volume conduction. In addition, exploratory **Phase-Amplitude Coupling (PAC)** analysis is included to examine cross-frequency interactions.

The workflow consists of EEG preprocessing, quality assessment, signal segmentation, connectivity estimation, statistical analysis, and visualization through connectivity matrices, power spectral density (PSD) plots, heatmaps, and brain network representations. These analyses enable a comprehensive comparison of functional brain organization across both study groups.

The primary objective of this repository is to provide a reproducible pipeline for EEG functional connectivity analysis and to explore potential neurophysiological biomarkers that may support future clinical assessment and neuroscience research.

## Highlights

- EEG functional connectivity analysis using MSC and wPLI
- Comparison of IDD and TDC brain networks
- Automated EEG preprocessing and quality assessment
- Multi-band analysis (Delta, Theta, Alpha, Beta, Gamma)
- Connectivity heatmaps and brain network visualizations
- Exploratory Phase-Amplitude Coupling (PAC) analysis
- Python implementation using MNE-Python and scientific computing libraries

## Methods

## Methods

The analysis pipeline follows a structured workflow for identifying EEG-based biomarkers from resting-state EEG recordings.

### 1. EEG Data Loading
- Load EEG recordings and subject information.
- Verify data integrity and channel configuration.

### 2. EEG Preprocessing
- Remove artifacts and noisy segments.
- Apply filtering and signal normalization.
- Perform quality assessment before analysis.

### 3. Time–Frequency Analysis
- Perform wavelet-based analysis to investigate frequency-specific brain activity.

### 4. Functional Connectivity Analysis
Two complementary connectivity measures are used:

- **Magnitude Squared Coherence (MSC):** Measures frequency-domain synchronization between EEG channels.
- **Weighted Phase Lag Index (wPLI):** Measures phase synchronization while minimizing the influence of volume conduction.

### 5. Network Analysis
- Construct functional brain connectivity matrices.
- Evaluate network characteristics using Small-Worldness and other graph-based measures.

### 6. Machine Learning
- Extract connectivity-based features.
- Train and evaluate machine learning models for distinguishing IDD and TDC participants.

### 7. Visualization
Generate:
- Connectivity heatmaps
- Brain connectivity matrices
- Power Spectral Density (PSD) plots
- Network visualizations

## Results

The analysis pipeline generates multiple visual and quantitative outputs to evaluate functional brain connectivity and identify potential EEG biomarkers.

### Generated Outputs

- Functional connectivity matrices using Magnitude Squared Coherence (MSC)
- Functional connectivity matrices using Weighted Phase Lag Index (wPLI)
- Power Spectral Density (PSD) analysis
- Time–frequency representations using Wavelet Transform
- Small-Worldness network analysis
- Connectivity heatmaps
- Brain network visualizations
- Machine learning predictions for classification

### Example Results

The `figures/` directory contains visualizations generated throughout the analysis, including:

- EEG connectivity heatmaps
- Brain connectivity networks
- PSD plots
- Wavelet analysis results
- Small-Worldness graphs

These visualizations support the comparison of functional connectivity patterns between Individuals with Intellectual Disabilities (IDD) and Typically Developing Controls (TDC).

## How to Run

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/EEG-Biomarkers.git
cd EEG-Biomarkers
```

### 2. Install the required packages

```bash
pip install -r requirements.txt
```

### 3. Open Jupyter Notebook

```bash
jupyter notebook
```

### 4. Run the notebooks in sequence

1. `01.ipynb`
2. `CFCnotebook.ipynb`
3. `wavelet.ipynb`
4. `Small-Worldness.ipynb`
5. `robustness.ipynb`
6. `ML_Apply.ipynb`

The notebooks should be executed in the above order, as each stage builds upon the outputs of the previous analysis.

## Repository Structure

## Repository Structure

```
EEG-Biomarkers/
│
├── README.md                 # Project documentation
├── LICENSE                   # MIT License
├── requirements.txt          # Required Python packages
├── .gitignore               # Files ignored by Git
│
├── notebooks/
│   ├── 01.ipynb
│   ├── CFCnotebook.ipynb
│   ├── wavelet.ipynb
│   ├── Small-Worldness.ipynb
│   ├── robustness.ipynb
│   └── ML_Apply.ipynb
│
├── figures/                 # Generated plots and visualizations
└── data/                    # Dataset location (not included)
```

## Future Work

*Coming soon.*

## Citation

If you use this repository in your research, please cite it appropriately.

## License

This project is licensed under the MIT License.

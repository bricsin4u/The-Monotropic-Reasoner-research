# The Monotropic Reasoner: Research Repository

This repository contains the dataset, simulation code, and validation results for the paper **"The Monotropic Reasoner: A Cybernetic Model of Neurodivergent Cognitive Efficiency"** by Igor S. Petrenko (2026).

## 1. Project Structure

The project is organized as follows:

*   `manuscript/`: Contains the full academic paper (`Autism_Cognitive_Model.md`) and the risk projection report.
*   `src/`: Python source code for simulations and data analysis.
    *   `Validation_Autism.py`: Monte Carlo simulation script.
    *   `analyze_real_data.py`: Big Five dataset analysis script.
*   `data/`: Contains the raw and processed datasets.
*   `figures/`: Generated charts and visualizations used in the paper.

## 2. Experimental Results

The research produced four key figures (located in `figures/`):

*   **Figure 1**: Rationality (G) distribution (Synthetic NT vs ND).
*   **Figure 2**: The "Meltdown Phase Transition" (Noise Impact).
*   **Figure 3**: Real-world G-score density estimation.
*   **Figure 4**: Theoretical Meltdown Rate comparison (90.1% vs 2.9%).

## 3. Replication Guide

To replicate the findings:

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run Synthetic Simulation**:
    ```bash
    cd src
    python Validation_Autism.py
    # Figures 1 & 2 will be saved to ../figures/
    ```

3.  **Run Real-World Analysis**:
    ```bash
    cd src
    python analyze_real_data.py
    # Figures 3 & 4 will be saved to ../figures/
    ```

## 4. Citation

Please cite this work using the metadata in `CITATION.cff` or:

> **Petrenko, I. S.** (2026). *The Monotropic Reasoner: A Cybernetic Model of Neurodivergent Cognitive Efficiency*. [Preprint].

## 5. Data Source

The real-world data used in this study is sourced from:
**Open-Source Psychometrics Project (2012)**. *Big Five Personality Test Data*. Retrieves from [https://openpsychometrics.org/_rawdata/](https://openpsychometrics.org/_rawdata/).


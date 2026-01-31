One-Bit Distributed Mean Estimation:
Adaptive and Non-Adaptive Schemes
================================================================================

This repository contains simulation code for evaluating adaptive and
non-adaptive one-bit communication schemes for distributed mean estimation.
The performance metric of interest is the Mean Squared Error (MSE).

The code supports multiple symmetric distributions and evaluates:
- Empirical MSE from simulations
- Theoretical upper bounds
- Theoretical lower bounds (adaptive, non-adaptive, and Fisher-information-based)

The results enable a direct comparison between adaptive and non-adaptive
schemes under strict communication constraints.

--------------------------------------------------------------------------------
Setup
--------------------------------------------------------------------------------

Install all required dependencies using:

    pip install -r requirements.txt

--------------------------------------------------------------------------------
Core Module
--------------------------------------------------------------------------------

- All_Schemes.py

This is the central module used by all experiments. It contains:
  * Distribution definitions (Gaussian, Logistic, Hyperbolic Secant, Sin2)
  * Sampling routines for location–scale families
  * Encoding and decoding rules for adaptive and non-adaptive schemes
  * Functions to compute theoretical MSE bounds

All experiment scripts import this file and therefore it must remain in the
root directory.

--------------------------------------------------------------------------------
Experiment Folders
--------------------------------------------------------------------------------

1. MSE_vs_Samples

This experiment studies the decay of MSE as the number of users (samples)
increases.

Files:
  - Experiment.py : runs simulations and saves results
  - Plot.py       : plots average-case and worst-case MSE curves

Upon execution, the script creates a directory named Avg_Worst_MSE_data
containing:
  * Worst_Case    – worst-case MSE across mean values
  * Average_Case  – average MSE across mean values
  * Benchmark     – theoretical lower bound data
  * Upper_Bound   – computed upper bound data

All results are stored as .pkl files (Python dictionaries), enabling fast
replotting without rerunning simulations.

Plots are saved as PDF files in:
  Worst_Average_MSE_Plots/

--------------------------------------------------------------------------------

2. MSE_vs_Mean

This experiment evaluates how the MSE varies with the true mean (location
parameter), while keeping the variance fixed.

Files:
  - Experiment.py : runs simulations over a grid of mean values
  - Plot.py       : plots MSE versus mean

Results are saved in:
  MSE_vs_mu_Data/

Plots are saved in:
  MSE_vs_mu_Plots/

--------------------------------------------------------------------------------

3. Adaptive_NonAdaptive_K

This experiment analyzes the effect of different (k1, k2) sample splits in
the adaptive scheme and compares them with the non-adaptive scheme.

Files:
  - Experiment.py : runs simulations for multiple (k1, k2) configurations
  - Plot.py       : generates comparison plots

Output folders:
  * Plots_Non-Adaptive – non-adaptive results
  * Plots_Adaptive     – adaptive results
  * Plots_Combined     – direct comparisons

All data are stored in .pkl format and figures are saved as PDFs.

--------------------------------------------------------------------------------

4. MSE_Beta

This experiment studies the dependence of theoretical constants on the
shape parameter (beta) of the generalized Gaussian distribution.

Files:
  - Experiment.py : computes constants for varying beta
  - Plot.py       : generates plots

Output:
  - constants_vs_beta.pdf
  - ratio_vs_beta.pdf

Saved in:
  Beta_Plots/

--------------------------------------------------------------------------------

5. MSE_Fisher_Bound

This folder compares the proposed adaptive and non-adaptive lower bounds
with the classical Fisher information lower bound in the unquantized setting.

Files:
  - fisher.py : computes Fisher-information-based lower bounds
  - plot.py   : visualizes and compares all three lower bounds

This experiment highlights the performance gap induced by one-bit
quantization relative to the ideal (no-quantization) case.

--------------------------------------------------------------------------------

6. SIN2_Distribution_Analysis

This folder verifies theoretical properties of the custom Sin2 distribution.

Files:
  - SIN2_data.py

The script checks:
  * normalization
  * variance
  * log-concavity
  * relevant theoretical constants

Results are printed to the terminal as a summary table. No files are saved.

--------------------------------------------------------------------------------
Running the Experiments
--------------------------------------------------------------------------------

Each experiment can be executed independently. Example:

    python MSE_vs_Samples/Experiment.py
    python MSE_vs_Samples/Plot.py

    python MSE_vs_Mean/Experiment.py
    python MSE_vs_Mean/Plot.py

    python Adaptive_NonAdaptive_K/Experiment.py
    python Adaptive_NonAdaptive_K/Plot.py

    python MSE_Beta/Experiment.py
    python MSE_Beta/Plot.py

    python SIN2_Distribution_Analysis/SIN2_data.py

All plots are saved in PDF format in their respective directories.

--------------------------------------------------------------------------------
Default Parameters
--------------------------------------------------------------------------------

- Number of trials        : 2000
- Maximum number of users : 40000
- Mean range              : [-2.5, 2.5]
- Standard deviation      : 2.0
- Distributions           : Gaussian, Logistic, Hyperbolic Secant, Sin2

All random seeds are fixed to ensure full reproducibility.
--------------------------------------------------------------------------------

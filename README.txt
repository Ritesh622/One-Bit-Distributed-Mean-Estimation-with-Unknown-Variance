One-Bit Estimation: Adaptive and Non-Adaptive Schemes
====================================================================================================

This code evaluates the performance of adaptive and non-adaptive one-bit
communication schemes for distributed mean estimation. It computes the
Mean Squared Error (MSE) for different source distributions, different
sample sizes, and various experiment configurations.

The results can be used to compare theoretical lower and upper bounds
with the simulated MSE for both adaptive and non-adaptive cases.

---------------------------------------------------------------------------------------------------
Setup
---------------------------------------------------------------------------------------------------

 Install all dependencies listed in the `requirements.txt` file as:

    pip install -r requirements.txt

---------------------------------------------------------------------------------------------------
Main File
---------------------------------------------------------------------------------------------------

- All_Schemes.py
  This is the main module used by all experiments.
  It contains:
    * Implementation of all distributions (Gaussian, Logistic,
      Hyperbolic Secant, and Sin2)
    * Sampling functions with location and scale parameters
    * Encoding and decoding rules for adaptive and non-adaptive schemes
    * Functions for theoretical bounds on MSE

All other scripts import this file, so it should be kept in the root
directory.

--------------------------------------------------------------------------------------------------
Folder Descriptions
--------------------------------------------------------------------------------------------------

1. MSE_vs_Samples
   This folder contains the main experiment that checks how MSE
   decreases as the number of users (samples) increases.

   - Experiment.py :  runs the simulation and saves the results.
   - Plot.py: plots the average and worst-case MSE curves.

   When Experiment.py is executed, it automatically creates a folder
   named Avg_Worst_MSE_data which includes several subfolders:
      * Worst_Case        – stores worst-case MSE results
      * Average_Case      – stores average-case MSE results
      * Benchmark         – contains theoretical lower bound data
      * Upper_Bound       – contains computed upper bound data

   Each file is saved in .pkl (pickle) format, which can be directly
   loaded later for plotting or further analysis.

   Example .pkl files:
      - gaussian_worst_case.pkl
      - logistic_average_case.pkl
      - hypsecant_nonadaptive_lb.pkl
      - sin2_nonadaptive_ub.pkl

   The files store results in the form of Python dictionaries, containing:
      * The list of total samples used in the experiment
      * Average and worst-case MSE values for adaptive and non-adaptive schemes
      * The best mean value where the worst-case occurs
      * Lower and upper bound constants

   These .pkl files make it easy to replot figures without re-running the entire simulation.

   After the data are generated, Plot.py uses these files to create
   PDF figures in the folder Worst_Average_MSE_Plots.

   Example plots:
      - MSE_vs_Samples_gaussian.pdf
      - MSE_vs_Samples_logistic.pdf

---------------------------------------------------------------------------------------------

2. MSE_vs_Mean
   This experiment checks how MSE changes with the true mean
   (location parameter) of the distribution while keeping variance fixed.

   - Experiment.py runs the simulation for different mean values.
   - Plot.py plots MSE versus mean.

   Results are saved as .pkl files under MSE_vs_mu_Data
   (automatically created).  
   The plots are saved in MSE_vs_mu_Plots.

----------------------------------------------------------------------------------------------

3. Adaptive_NonAdaptive_K
   This folder compares how different (k1, k2) split ratios affect
   the adaptive scheme, and how it performs compared to the
   non-adaptive scheme.

   - Experiment.py runs simulations for multiple k1, k2 settings.
   - Plot.py produces plots showing both adaptive and non-adaptive results.

   Results are stored in .pkl format and figures are saved as PDFs in:
      * Plots_Non-Adaptive – shows non-adaptive results for each setting
      * Plots_Adaptive – shows adaptive results for each setting
      * Plots_Combined – shows adaptive and non-adaptive comparisons

-----------------------------------------------------------------------------------------------

4. MSE_Beta
   This experiment studies how the theoretical constants depend on the
   shape parameter (beta) in the generalized Gaussian distribution.

   - Experiment.py calculates constants for different beta values.
   - Plot.py creates plots for constants versus beta.

   Output plots:
      - constants_vs_beta.pdf
      - ratio_vs_beta.pdf

   Saved inside the folder Beta_Plots.

------------------------------------------------------------------------------------------------
5. MSE_Fisher_Bound

This folder compares the proposed adaptive and non-adaptive lower bounds with the classical lower bound derived using 
Fisher information in the no-quantization setting.

  -fisher.py is the main script used to generate the corresponding lower-bound data.

  -plot.py can be used to visualize and compare the lower bounds on the MSE for all three cases.

------------------------------------------------------------------------------------------------

6. SIN2_Distribution_Analysis
   This folder verifies the mathematical properties of the custom Sin2
   distribution used in the paper.

   - SIN2_data.py checks normalization, variance, log-concavity,
     and theoretical constants for different p values.

   It prints a summary table in the terminal showing key metrics such as:
      - normalization constant
      - variance
      - minimum phi''(x)
      - computed constants and ratios

   This script does not generate or save data files.



-----------------------------------------------------------------------------------------------
Running the Scripts
-----------------------------------------------------------------------------------------------

Each experiment is independent and can be executed separately.
For example:

    python MSE_vs_Samples/Experiment.py
    python MSE_vs_Samples/Plot.py

    python MSE_vs_Mean/Experiment.py
    python MSE_vs_Mean/Plot.py

    python Adaptive_NonAdaptive_K/Experiment.py
    python Adaptive_NonAdaptive_K/Plot.py

    python MSE_Beta/Experiment.py
    python MSE_Beta/Plot.py

    python SIN2_Distribution_Analysis/SIN2_data.py

All generated plots are stored in their respective folders in PDF format.

----------------------------------------------------------------------------------------------
Important Parameters
----------------------------------------------------------------------------------------------

- Number of trials: 2000
- Maximum number of samples: 40000
- Mean range: -2.5 to 2.5
- Standard deviation (scale): 2.0
- Distributions used: Gaussian, Logistic, Hyperbolic Secant, Sin2

All random seeds are fixed to ensure that the results are exactly reproducible.
-----------------------------------------------------------------------------------------------

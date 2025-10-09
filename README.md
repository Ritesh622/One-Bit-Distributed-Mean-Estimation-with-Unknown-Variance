# One-Bit-Distributed-Mean-Estimation-with-Unknown-Variance

This repository contains the official code for the paper:
“One-Bit Distributed Mean Estimation with Unknown Variance” by Ritesh Kumar and Shashank Vatedka
In this paper, We design non-adaptive and adaptive one-bit protocols for distributions belonging to scale–location families and derive exact asymptotic bounds on the achievable Mean Squared Error (MSE).
Our results establish a strict performance gap between adaptive and non-adaptive schemes for a broad class of symmetric, strictly log-concave distributions (including Gaussian, logistic, and hyperbolic secant families).
There are five main folders:

  * MSE_vs_Samples: This folder contains the main experiment that checks how MSE decreases as the number of users (samples) increases.
  * MSE_vs_Mean: This experiment checks how MSE changes with the true mean (location parameter) of the distribution while keeping variance fixed.
  
  * Adaptive_NonAdaptive_K: This folder compares how different (k1, k2) split ratios affect the adaptive scheme, and how it performs compared to the non-adaptive scheme.
  
  * MSE_Beta: This experiment studies how the theoretical constants depend on the shape parameter (beta) in the generalized Gaussian distribution.

  * SIN2_Distribution_Analysis: This folder verifies the mathematical properties of the custom Sin2 distribution used in the paper.
  

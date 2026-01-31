One-Bit Distributed Mean Estimation with Unknown Variance
=========================================================
This repository contains the official code for the paper: “One-Bit Distributed Mean Estimation with Unknown Variance” by Ritesh Kumar and Shashank Vatedka.

In this paper, we design non-adaptive and adaptive one-bit communication protocols for distributed mean estimation for distributions belonging to scale–location families with unknown variance. We derive exact asymptotic bounds on the achievable Mean Squared Error (MSE).

Our results establish a strict performance gap between adaptive and non-adaptive schemes for a broad class of symmetric, strictly log-concave distributions, including Gaussian, logistic, and hyperbolic secant families.

There are six main folders:

  * MSE_vs_Samples:  This folder contains the main experiment that checks how MSE decreases as the number of users (samples) increases. It compares empirical MSE with theoretical upper and lower bounds for both adaptive and non-adaptive schemes.

  * MSE_vs_Mean: This experiment checks how MSE changes with the true mean (location parameter) of the distribution while keeping variance fixed.

  * Adaptive_NonAdaptive_K: This folder compares how different (k1, k2) split ratios affect the adaptive scheme and how it performs compared to the non-adaptive scheme.

  * MSE_Beta: This experiment studies how the theoretical constants depend on the shape parameter (beta) in the generalized Gaussian distribution.

 * MSE_Fisher_Bound: This folder compares the proposed adaptive and non-adaptive lower bounds with the classical lower bound derived using Fisher information in the no-quantization setting.
                     This experiment highlights the performance loss due to one-  bit quantization.

  * SIN2_Distribution_Analysis:  This folder verifies the mathematical properties of the custom  Sin2 distribution used in the paper, including normalization, variance, log-concavity, and related theoretical constants.


# Dimensionality Reduction for Heavy-Tailed Data

A comparative analysis of Robust PCA techniques against heavy-tailed distributions and gross corruption. This project benchmarks Standard PCA, Principal Component Pursuit (RPCA), Tyler's M-Estimator, and Robust Bayesian PCA across synthetic alpha-stable data, financial time series, and corrupted image datasets.

## Project Structure

* `rpca_tests.py`: Main file for running tests. Run this one.
  
* `PCA_functions_generation.py`: Module for generating synthetic heavy-tailed data (Sub-Gaussian, Student-t, Pareto), fetching real financial data from `real_stock1.csv` or 'real_stock2.csv`, and fetching real image data from sklearn datasets.
* `PCA_functions_component.py`: Helpful functions to fetch ground truth vectors and calculate angle errors
* `PCA_functions_plotting.py`: Helpful plotting functions
  
* `image_test.py`: Contains the main function to do image tests.
* `PCA_classes.py`: Implementations of dimensionality reduction techniques (PCP, L1-PCA, Bayesian PCA).
  
* `parser.py`: Parser file.
* `requirements.txt`: List of dependencies.

## Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/NourGRaffoul/PCA_heavy_tails.git] (https://github.com/NourGRaffoul/PCA_heavy_tails.git)
    cd PCA_heavy_tails
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

	NOTE: preferably run on python 3.11. 

## Usage

Some examples:

```bash
python rpca_tests.py --covariance 1 0.3 0.3 4 --plot True
```
```bash
python rpca_tests.py --mode student_t --dim 10 --n_samples 5000 --alphas 1.1 1.3 1.6 1.8
```
```bash
python rpca_tests.py --mode image1 --contamination 0.6
```
```bash
python rpca_tests.py --mode real_stock2 --dim 30 --n_samples 1000
```

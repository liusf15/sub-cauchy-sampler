# sub-cauchy-sampler

Code for reproducing the numerical experiments in *Sub-Cauchy Sampling: Escaping the Dark Side of the Moon* by Sebastiano Grazzi, Sifan Liu, Gareth O. Roberts, and Jun Yang.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/liusf15/sub-cauchy-sampler.git
   cd sub-cauchy-sampler
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv_scp
   . venv_scp/bin/activate 
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```


## Experiments and notebooks

Run the skew-t comparison across all samplers:

```bash
python -m experiments.run_skewt --affine scalar
```

Run the logistic regression comparison across all samplers:

```bash
python -m experiments/run_logistic --affine covariance 
```

Example code for multivariate probit regression in [experiments/multivariate_probit.ipynb](experiments/multivariate_probit.ipynb).

Example code for logistic regression with horseshoe prior in [experiments/logistic_regression_horseshoe.ipynb](experiments/logistic_regression_horseshoe.ipynb).

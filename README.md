# sub-cauchy-sampler

Code accompanying [*Sub-Cauchy Sampling: Escaping the Dark Side of the Moon*](https://arxiv.org/abs/2601.11066) by Sebastiano Grazzi, Sifan Liu, Gareth O. Roberts, and Jun Yang.

The repository is organized as follows:

- `src/`: implementation of the sub-Cauchy projection sampler and comparison samplers.
- `experiments/skew_t/`: skew-t experiment runners and plotting notebooks.
- `experiments/regression/`: logistic/robit regression runners for Student-t and horseshoe priors.
- `results/`: generated raw CSV/meta outputs. The directory is ignored by git.
- `plots/`: generated plots included in the paper.

## Installation

Use Python 3.10 or newer.

```bash
git clone https://github.com/liusf15/sub-cauchy-sampler.git
cd sub-cauchy-sampler

python3 -m venv venv_scp
. venv_scp/bin/activate
python3 -m pip install -r requirements.txt
```

## Reproducing Experiments

### Multivariate skew-t

```bash

python3 -m experiments.skew_t.run_skewt --seeds 0:10 

python3 -m experiments.skew_t.plot_qq_grid 
```

Estimating tail probabilities

```bash
python3 -m experiments.skew_t.tail_variance --date skewt_tail --nseeds 20 
```

### Robust regression

#### Logistic link, student t prior

```bash

python3 -m experiments.regression.logistic_student_t \
  --methods Gibbs HMC IMH SCP --seeds 0:10

python3 -m experiments.regression.plot_qq_grid --figure logistic_t
```

#### Robit link, student t prior

```bash

python3 -m experiments.regression.robit_student_t \
  --methods Gibbs HMC IMH SCP --seeds 0:10

python3 -m experiments.regression.plot_qq_grid --figure robit_t
```

#### Logistic link, horseshoe prior

```bash
python -m experiments.regression.logistic_horseshoe \
--methods Gibbs HMC IMH SCS --seeds 0:10

python -m experiments.regression.plot_logistic_horseshoe_noncentered_beta1_qq
```

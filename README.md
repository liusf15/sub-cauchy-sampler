# sub-cauchy-sampler


## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/liusf15/sub-cauchy-sampler.git
   cd sub-cauchy-sampler
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv scp_venv
   . scp_venv/bin/activate 
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

See example usage in the [notebook](experiments/examples.ipynb).

To run experiments,

```bash
python -m experiments.run_experiment --target skewt --d 10 --latitude 1.5 --nsample 1000_000
```

# InsuranceClaimStatusPredictor

Binary classifier for predicting insurance claim status from historical claims data.

## Setup

From the project root, create and activate a Python environment:

```bash
cd  InsuranceClaimStatusPredictor
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Because the project imports modules from the `src` directory, add it to `PYTHONPATH` before running any scripts:

```bash
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
```

To make this persistent in your shell, add the same line to your shell profile, for example:

```bash
echo 'export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"' >> ~/.bashrc
source ~/.bashrc
```

## Example usage

Train the model:

```bash
python src/train/train.py --data_path
```

Run inference on a dataset:

```bash
python src/inference/inference.py \
  --model_path <path to model.pkl> \
  --data_path <dataset_path_to_current_claims.csv> (eg:dataset/current_claims.csv ) \
  --top_frac 0.25
```

## Write-up

A detailed write-up describing the approach is available in [Claim_Denial_Prediction.pdf](Claim_Denial_Prediction.pdf).

## Requirements

The required Python packages are listed in `requirements.txt`.

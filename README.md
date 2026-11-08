# recommender-systems-group-23
Recommender System Project Group 23
## Requirements

- Python 3.12.3

## Setup

Create and activate a virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

> **Wrong Python version?** If `pip install -e .` fails with
> `ERROR: Package 'rec-system' requires a different Python: 3.14.3 not in '<3.13,>=3.12'`,
> your `.venv` was created with a different Python (often plain `python3`).
> Check with `python --version` inside the venv. If Python 3.12 is not installed,
> get it with `brew install python@3.12` or let [uv](https://docs.astral.sh/uv/) download it,
> then recreate the venv:
>
> ```bash
> deactivate
> rm -rf .venv
> uv venv --python 3.12.3 .venv   # or: python3.12 -m venv .venv
> source .venv/bin/activate
> ```

Install the dependencies and the project:

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

## Data

This project uses MovieLens 1M from GroupLens: https://grouplens.org/datasets/movielens/1m/

The dataset is not included in this repository. To download it:

```bash
mkdir -p data
curl -L -o data/ml-1m.zip https://files.grouplens.org/datasets/movielens/ml-1m.zip
unzip data/ml-1m.zip -d data
```

## Running the project

> Not implemented yet. This command will reproduce all results and figures once the pipeline is complete.

```bash
python run_all.py --data-dir data/ml-1m
```
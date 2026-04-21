# IPD-LLM

Iterated Prisoner's Dilemma simulation supporting a variety of Agent algorithms
including Ollama-served LLMs.

## Installation

```bash
# Install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python cli.py -h
python cli.py 40 --agent-fracs Sucker 0.3 Mean 0.3 TFT 0.4
```

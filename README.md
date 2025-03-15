# pytorch-plantclef
PyTorch webinar on using DINOv2 for plant species classification.


## Quickstart

Install `uv` as the package manager for the project:
- Follow the `uv` [installation instructions](https://docs.astral.sh/uv/getting-started/installation/).

Create a virtual environment:
```bash
uv venv venv
```

Activate the virtual environment:
```bash
source venv/bin/activate
```

Install the pre-commit hooks for formatting code:

```bash
pre-commit install
```

Install the requirement packages to the `venv` virtual environment:

```bash
uv pip install -r requirements.txt
```

Install the package in "editable" mode, which means changes to the Python files will be immediately available without needing to reinstall the package.

```bash
uv pip install -e .
```

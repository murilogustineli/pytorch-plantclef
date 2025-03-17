# pytorch-plantclef

PyTorch webinar on using DINOv2 for plant species classification.
This is part of the [PlantCLEF @ LifeCLEF & CVPR-FGVC](https://www.kaggle.com/competitions/plantclef-2025) competition on Kaggle.

## Quickstart

Clone the [**`pytorch-plantclef`**](https://github.com/murilogustineli/pytorch-plantclef) repo:

- Using HTTPS (recommended when using Intel Tiber AI Cloud):

```bash
git clone https://github.com/murilogustineli/pytorch-plantclef.git
```

- Using SSH:

```bash
git clone git@github.com:murilogustineli/pytorch-plantclef.git
```

Navigate to the project directory:

```bash
cd pytorch-plantclef
```

Install `uv` as the package manager for the project. Follow the `uv` [installation instructions](https://docs.astral.sh/uv/getting-started/installation/) for macOS, Linux, and Windows.

If running on [Intel Tiber AI Cloud](https://ai.cloud.intel.com/), install `uv` as the following (also works for macOS and Linux):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Add it to PATH:

```bash
source $HOME/.local/bin/env
```

Check `uv` was installed correctly:

```bash
uv --version
```

Create a virtual environment:

```bash
uv venv venv
```

Activate the virtual environment:

```bash
source venv/bin/activate
```

Install the requirement packages to the `venv` virtual environment:

```bash
uv pip install -r requirements.txt
```

Install the package in "editable" mode, which means changes to the Python files will be immediately available without needing to reinstall the package.

```bash
uv pip install -e .
```

**[OPTIONAL]**: Install the pre-commit hooks for formatting code:

```bash
pre-commit install
```

Download the dataset and fine-tuned ViT model:

```bash
bash scripts/download_data_model.sh
```

Test the environment by running the `pystest`:

```bash
pytest -vv -s tests/test_embed.py
```

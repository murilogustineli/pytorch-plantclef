# pytorch-plantclef

![plantclef-2025-banner](https://www.imageclef.org/system/files/new_banner_plantclef2025.png)

PyTorch webinar on using the **DINOv2** model for multi-label plant species classification.
This project is part of the [**PlantCLEF @ LifeCLEF & CVPR-FGVC**](https://www.kaggle.com/competitions/plantclef-2025) competition on Kaggle.

## Quickstart Guide

### 1. Clone the repository

First, clone the [**`pytorch-plantclef`**](https://github.com/murilogustineli/pytorch-plantclef) repo:

⚠️ **Using HTTPS** _(Recommended for Intel Tiber AI Cloud)_:

```bash
git clone https://github.com/murilogustineli/pytorch-plantclef.git
```

**Using SSH**:

```bash
git clone git@github.com:murilogustineli/pytorch-plantclef.git
```

Navigate to the project directory:

```bash
cd pytorch-plantclef
```

### 2. Install `uv` (Fast Package Manager)

Install `uv` as the package manager for the project. Follow the `uv` [installation instructions](https://docs.astral.sh/uv/getting-started/installation/) for macOS, Linux, and Windows.

If running on [Intel Tiber AI Cloud](https://ai.cloud.intel.com/), install `uv` as the following (also works for macOS and Linux):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Add it to `PATH`:

```bash
source $HOME/.local/bin/env
```

Check `uv` installation:

```bash
uv --version
```

### 3. Create a Virtual Environment

Create the virtual environment:

```bash
uv venv venv
```

Activate the virtual environment:

```bash
source venv/bin/activate
```

### 4. Install Dependencies and Set Up the Project

Install the `plantclef` package in **_editable mode_**, which means changes to the Python files will be immediately available without needing to reinstall the package.

Install all dependencies from `requirements.txt` to the `venv` virtual environment:

```bash
uv pip install -e .
```

This command does two things:

1. Installs all dependencies listed in `requirements.txt`.
2. Sets up `plantclef` as an **editable package** inside the virtual environment.

#### **[OPTIONAL] Set Up Pre-Commit Hooks for Code Formatting:**

To ensure code follows best practices, install `pre-commit`:

```bash
pre-commit install
```

This automatically formats and checks your code before every commit.

### 5. Download Dataset & Fine-Tuned ViT Model

Run the following script to download:

- **Dataset** (`data/parquet/dataset_name`)
- **Fine-Tuned DINOv2 Model** (`model/pretrained_models/model_name`)

```bash
bash scripts/download_data_model.sh
```

This script will:

- Download the dataset & model from Google Drive.
- Extract the `.zip` files into their respective directories.
- Remove the original `.zip` files to save space.

### 6. Run tests to verify setup

After downloading the data and fine-tuned model, we can ensure everything is working correctly by running the following `pystest`:

```bash
pytest -vv -s tests/test_model.py
```

This test ensures that:

- The **virtual environment** is correctly set up.
- The **DINOv2 model** is correctly loaded.
- Image embeddings are generated without errors.

If you're **running locally**, you should be good to go! If you're running on **Intel Tiber AI Cloud**, follow the setup below.

## Intel Tiber AI Cloud Setup

⚠️ The **Jupyter** and **terminal** environments on ITAC are **NOT** synced. This means that installing packages or setting environment variables in one will not automatically apply to the other.

To ensure proper Intel GPU (`xpu`) access, follow these steps:

1. **Open the notebook:** Open the jupyter notebook `notebooks/setup_itac.ipynb`.
2. **Run cells sequentially:** Go through the notebook step by step.
3. **Restart the Kernel when required:** Running the cell `exit()` will restart the jupyter kernel to apply the installations.
4. **Verify that the Intel GPU (`xpu`) is being used:** At the end of the notebook execution, check the PyTorch version and device are correct. The expect output if Intel GPU is enabled is:

   ```
   PyTorch Version: 2.5.1+cxx11.abi
   Using device: xpu
   ```

   If you see `Using device: cpu`, the setup did not correctly enable the Intel GPU—retry running the setup notebook.

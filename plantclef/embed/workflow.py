import torch
import numpy as np
import pandas as pd
from .transform import PlantDataset, DINOv2LightningModel
from torch.utils.data import DataLoader
from tqdm import tqdm


def extract_embeddings(
    pandas_df: pd.DataFrame,
    batch_size: int = 32,
    use_grid: bool = False,
    grid_size: int = 4,
    cpu_count: int = 4,
) -> np.ndarray:
    """Extract embeddings for images in a Pandas DataFrame using PyTorch Lightning."""

    # initialize model
    model = DINOv2LightningModel()

    # create Dataset
    dataset = PlantDataset(
        pandas_df,
        model.transform,
        use_grid=use_grid,
        grid_size=grid_size,
    )
    # create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cpu_count,
        collate_fn=lambda batch: (
            torch.cat(batch, dim=0) if use_grid else torch.stack(batch)
        ),
    )

    # run inference and collect embeddings with tqdm progress bar
    all_embeddings = []
    for batch in tqdm(dataloader, desc="Extracting embeddings", unit="batch"):
        embeddings = model(batch)
        all_embeddings.append(embeddings.cpu().numpy())

    return np.vstack(all_embeddings)  # combine all embeddings into a single array

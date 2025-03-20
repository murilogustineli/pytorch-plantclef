import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from plantclef.torch.data import (
    PlantDataset,
    PlantDataModule,
    custom_collate_fn_partial,
)
from plantclef.torch.model import DINOv2LightningModel
from plantclef.config import get_device
from torch.utils.data import DataLoader
from tqdm import tqdm


def inference_pipeline(
    pandas_df: pd.DataFrame,
    batch_size: int = 32,
    use_grid: bool = False,
    grid_size: int = 4,
    cpu_count: int = 4,
    top_k: int = 10,
) -> np.ndarray:
    """Pipeline to extract embeddings and top-K logits using PyTorch Lightning."""

    # initialize model
    model = DINOv2LightningModel(top_k=top_k)

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
        collate_fn=custom_collate_fn_partial(use_grid),  # pickle-friendly collate_fn
    )

    # run inference and collect embeddings with tqdm progress bar
    all_embeddings = []
    all_logits = []
    for batch in tqdm(
        dataloader, desc="Extracting embeddings and logits", unit="batch"
    ):
        embeddings, logits = model(batch)  # forward pass

        if use_grid:
            B = batch.shape[0]  # number of images in the batch
            G = grid_size**2  # number of tiles per image
            embeddings = embeddings.view(B, G, -1)  # flatten tiles into single tensor

        all_embeddings.append(embeddings.cpu().numpy())
        all_logits.append(logits.cpu().numpy())

    # combine all embeddings into a single array
    embeddings_stack = np.vstack(all_embeddings)
    logits_stack = np.vstack(all_logits)

    return embeddings_stack, logits_stack


def trainer_pipeline(
    pandas_df: pd.DataFrame,
    batch_size: int = 32,
    use_grid: bool = False,
    grid_size: int = 4,
    cpu_count: int = 4,
    top_k: int = 5,
):
    """Pipeline to extract embeddings and top-k logits using PyTorch Lightning."""

    # initialize DataModule
    data_module = PlantDataModule(
        pandas_df,
        batch_size=batch_size,
        use_grid=use_grid,
        grid_size=grid_size,
        num_workers=cpu_count,
    )

    # initialize Model
    model = DINOv2LightningModel(top_k=top_k)

    # define Trainer (inference mode)
    trainer = pl.Trainer(
        accelerator=get_device(),
        devices=1,
        enable_progress_bar=True,
    )

    # run Inference
    predictions = trainer.predict(model, datamodule=data_module)

    # unpack predictions: List[Tuple[embeddings, logits]]
    embeddings = torch.cat([batch[0] for batch in predictions], dim=0)
    logits = torch.cat([batch[1] for batch in predictions], dim=0)

    return embeddings, logits

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


def torch_pipeline(
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
        embeddings, logits = model.predict_step(
            batch, batch_idx=0
        )  # batch: List[Tuple[embeddings, logits]]
        all_embeddings.append(embeddings)  # keep embeddings as tensors
        reshaped_logits = [
            logits[i : i + grid_size**2] for i in range(0, len(logits), grid_size**2)
        ]
        all_logits.extend(reshaped_logits)  # preserve batch structure

    # convert embeddings to tensor
    embeddings = torch.cat(all_embeddings, dim=0)  # shape: [len(df), grid_size**2, 768]

    if use_grid:
        embeddings = embeddings.view(-1, grid_size**2, 768)
    else:
        embeddings = embeddings.view(-1, 1, 768)

    return embeddings, all_logits


def pl_trainer_pipeline(
    pandas_df: pd.DataFrame,
    batch_size: int = 32,
    use_grid: bool = False,
    grid_size: int = 1,
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

    all_embeddings = []
    all_logits = []
    for batch in predictions:
        embed_batch, logits_batch = batch  # batch: List[Tuple[embeddings, logits]]
        all_embeddings.append(embed_batch)  # keep embeddings as tensors
        reshaped_logits = [
            logits_batch[i : i + grid_size**2]
            for i in range(0, len(logits_batch), grid_size**2)
        ]
        all_logits.extend(reshaped_logits)  # preserve batch structure

    # convert embeddings to tensor
    embeddings = torch.cat(all_embeddings, dim=0)  # shape: [len(df), grid_size**2, 768]

    if use_grid:
        embeddings = embeddings.view(-1, grid_size**2, 768)
    else:
        embeddings = embeddings.view(-1, 1, 768)

    return embeddings, all_logits

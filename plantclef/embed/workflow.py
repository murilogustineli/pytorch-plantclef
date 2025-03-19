import numpy as np
import pandas as pd
from plantclef.torch.data import PlantDataset, custom_collate_fn_partial
from plantclef.torch.model import DINOv2LightningModel
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

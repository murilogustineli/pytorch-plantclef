import torch
import pytest

from torch.utils.data import DataLoader
from plantclef.embed.transform import DINOv2LightningModel, PlantDataset
from plantclef.embed.workflow import custom_collate_fn_partial
from plantclef.model_setup import setup_fine_tuned_model


@pytest.mark.parametrize(
    "model_name, expected_dim, use_grid, grid_size",
    [
        ("vit_base_patch14_reg4_dinov2.lvd142m", 768, False, 2),  # No grid
        ("vit_base_patch14_reg4_dinov2.lvd142m", 768, True, 2),  # Grid size 2
    ],
)
def test_finetuned_dinov2(
    pandas_df,
    model_name,
    expected_dim,
    use_grid,
    grid_size,
):
    model = DINOv2LightningModel(
        model_path=setup_fine_tuned_model(),
        model_name=model_name,
    )

    # create PlantDataset and DataLoader
    dataset = PlantDataset(
        pandas_df,
        transform=model.transform,
        use_grid=use_grid,
        grid_size=grid_size,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        collate_fn=custom_collate_fn_partial(use_grid),  # pickle-friendly collate_fn
    )

    # extract embeddings
    for batch in dataloader:
        embeddings = model(batch)  # forward pass

        if use_grid:
            B = batch.shape[0]  # number of images in the batch
            G = grid_size**2  # number of tiles per image
            embeddings = embeddings.view(B, G, -1)  # flatten tiles into single tensor

        assert isinstance(embeddings, torch.Tensor)
        expected_shape = (grid_size**2, expected_dim) if use_grid else (1, expected_dim)
        if use_grid:
            assert embeddings[0].shape == expected_shape
        else:
            assert embeddings.shape == expected_shape
        assert all(isinstance(x.item(), float) for x in embeddings.flatten())

import torch
import pytest
from torch.utils.data import DataLoader
from plantclef.torch.model import DINOv2LightningModel
from plantclef.torch.data import PlantDataset, custom_collate_fn_partial
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
    cpu_count=1,
    batch_size=1,
    top_k=10,
):
    model = DINOv2LightningModel(
        model_path=setup_fine_tuned_model(),
        model_name=model_name,
        top_k=top_k,
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
        batch_size=batch_size,
        shuffle=False,
        num_workers=cpu_count,
        collate_fn=custom_collate_fn_partial(use_grid),  # pickle-friendly collate_fn
    )

    for batch in dataloader:
        embeddings, logits = model(batch)  # forward pass

        if use_grid:
            B = batch.shape[0]  # number of images in the batch
            G = grid_size**2  # number of tiles per image
            embeddings = embeddings.view(B, G, -1)  # flatten tiles into single tensor

        assert isinstance(embeddings, torch.Tensor)
        assert all(isinstance(x.item(), float) for x in embeddings.flatten())
        expected_shape = (grid_size**2, expected_dim) if use_grid else (1, expected_dim)
        if use_grid:
            assert embeddings[0].shape == expected_shape
        else:
            assert embeddings.shape == expected_shape

        # check logits
        assert isinstance(logits, torch.Tensor)
        assert all(isinstance(x.item(), float) for x in logits.flatten())
        if use_grid:
            assert logits.shape == (grid_size**2, model.num_classes)
        else:
            assert logits.shape == (batch_size, model.num_classes)

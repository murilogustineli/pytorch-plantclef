import io
import torch
import pytest
import pandas as pd

from PIL import Image
from functools import partial
from torch.utils.data import DataLoader
from plantclef.embed.transform import DINOv2LightningModel, PlantDataset
from plantclef.embed.workflow import custom_collate_fn
from plantclef.model_setup import setup_fine_tuned_model


@pytest.fixture
def pandas_df():
    # generate a small dummy image(RGB, 32X32) for testing
    img = Image.new("RGB", (32, 32), color="blue")
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG")
    img_bytes = img_bytes.getvalue()
    data = {"data": [img_bytes, img_bytes]}

    return pd.DataFrame(data)


@pytest.mark.parametrize(
    "use_grid, grid_size",
    [
        (False, 2),  # No grid
        (True, 2),  # Grid size 2
    ],
)
def test_plant_dataset(pandas_df, use_grid, grid_size):
    dataset = PlantDataset(
        pandas_df,
        transform=None,
        use_grid=use_grid,
        grid_size=grid_size,
    )
    assert len(dataset) == 2
    sample_data = dataset[0]
    assert isinstance(sample_data, torch.Tensor)
    expected_shape = (
        (grid_size**2, *sample_data.shape[1:]) if use_grid else sample_data.shape
    )
    assert sample_data.shape == expected_shape


def custom_collate_fn_partial(use_grid):
    """Returns a pickle-friendly collate function with the `use_grid` flag."""
    return partial(custom_collate_fn, use_grid=use_grid)


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

        assert isinstance(embeddings, torch.Tensor)
        expected_shape = (grid_size**2, expected_dim) if use_grid else (1, expected_dim)
        assert embeddings.shape == expected_shape
        assert all(isinstance(x.item(), float) for x in embeddings.flatten())

import io
import torch
import pytest
import pandas as pd

from PIL import Image
from torch.utils.data import DataLoader
from plantclef.embed.transform import DINOv2LightningModel, PlantDataset
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
    if use_grid:
        assert isinstance(dataset[0], list)
        assert len(dataset[0]) == grid_size**2
        assert isinstance(dataset[0][0], Image.Image)
    else:  # no grid
        assert isinstance(dataset[0], Image.Image)


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
        collate_fn=lambda batch: (
            torch.cat(batch, dim=0) if use_grid else torch.stack(batch)
        ),
    )

    # extract embeddings
    for batch in dataloader:
        embeddings = model(batch)  # forward pass
        assert embeddings.shape == (1, expected_dim)
        if use_grid:
            assert isinstance(embeddings, list)
            assert all(isinstance(x, torch.Tensor) for x in embeddings)
            assert all(
                isinstance(y.item(), float) for x in embeddings for y in x.flatten()
            )
        else:  # no grid
            assert isinstance(embeddings, torch.Tensor)
            assert all(isinstance(x.item(), float) for x in embeddings.flatten())

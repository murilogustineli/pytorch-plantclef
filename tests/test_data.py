import torch
import pytest
from torch.utils.data import DataLoader
from plantclef.torch.data import PlantDataset, PlantDataModule


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


@pytest.mark.parametrize(
    "use_grid, grid_size, batch_size",
    [
        (False, 2, 1),  # No grid
        (True, 2, 1),  # Grid size 2
    ],
)
def test_plant_datamodule(pandas_df, use_grid, grid_size, batch_size):
    data_module = PlantDataModule(
        pandas_df,
        batch_size=batch_size,
        use_grid=use_grid,
        grid_size=grid_size,
        num_workers=1,
    )
    data_module.setup(stage="predict")
    # check if predict_dataloader is a valid DataLoader
    dataloader = data_module.predict_dataloader()
    assert isinstance(dataloader, DataLoader)

    # check if DataLoader outputs correct batch format
    batch = next(iter(dataloader))
    assert isinstance(batch, torch.Tensor)

    # check batch shape
    if use_grid:
        B, G, C, H, W = batch.shape
        assert B == batch_size
        assert G == grid_size**2  # grid should have (grid_size x grid_size) patches
    else:
        B, C, H, W = batch.shape
        assert B == batch_size  # single image per batch

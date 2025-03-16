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


def test_plant_dataset(pandas_df):
    dataset = PlantDataset(pandas_df, transform=None)
    assert len(dataset) == 2
    assert isinstance(dataset[0], Image.Image)


@pytest.mark.parametrize(
    "model_name,expected_dim",
    [
        ("vit_base_patch14_reg4_dinov2.lvd142m", 768),  # Adjust output dim if needed
    ],
)
def test_finetuned_dinov2(pandas_df, model_name, expected_dim):
    model = DINOv2LightningModel(
        model_path=setup_fine_tuned_model(),
        model_name=model_name,
    )

    # create PlantDataset and DataLoader
    dataset = PlantDataset(pandas_df, transform=model.transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # extract embeddings
    for batch in dataloader:
        embeddings = model(batch)  # forward pass
        assert embeddings.shape == (1, expected_dim)
        assert isinstance(embeddings, torch.Tensor)
        assert all(isinstance(x.item(), float) for x in embeddings.flatten())

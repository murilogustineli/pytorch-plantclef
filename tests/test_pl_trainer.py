import torch
import pytest
import pytorch_lightning as pl
from plantclef.torch.model import DINOv2LightningModel
from plantclef.torch.data import PlantDataModule
from plantclef.config import get_device


@pytest.mark.parametrize(
    "model_name, expected_dim, use_grid, grid_size",
    [
        ("vit_base_patch14_reg4_dinov2.lvd142m", 768, False, 2),  # No grid
        # ("vit_base_patch14_reg4_dinov2.lvd142m", 768, True, 2),  # Grid size 2
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
    # initialize DataModule
    data_module = PlantDataModule(
        pandas_df,
        batch_size=batch_size,
        use_grid=use_grid,
        grid_size=grid_size,
        num_workers=cpu_count,
    )

    # initialize Model
    model = DINOv2LightningModel(
        model_name=model_name,
        top_k=top_k,
    )

    # define Trainer (inference mode)
    trainer = pl.Trainer(
        accelerator=get_device(),
        devices=1,
        enable_progress_bar=True,
    )

    # run Inference
    predictions = trainer.predict(
        model, datamodule=data_module
    )  # List[Tuple[embeddings, logits]]

    # extract embeddings and logits from the predictions list
    embeddings = torch.cat([batch[0] for batch in predictions], dim=0)
    logits = torch.cat([batch[1] for batch in predictions], dim=0)

    assert isinstance(embeddings, torch.Tensor)
    assert all(isinstance(x.item(), float) for x in embeddings.flatten())

    # check total number of samples matches dataset size
    assert embeddings.shape[0] == len(pandas_df)  # Ensure all samples processed
    assert logits.shape[0] == len(pandas_df)

    # ensure embedding feature dimension is correct
    assert embeddings.shape[1] == expected_dim

    # ensure logits have correct class dimension
    assert logits.shape[1] == model.num_classes

    # expected_shape = (grid_size**2, expected_dim) if use_grid else (1, expected_dim)
    # if use_grid:
    #     assert embeddings[0].shape == expected_shape
    #     assert logits[0].shape == expected_shape
    # else:
    #     assert embeddings[0].shape == expected_dim
    #     assert logits[0].shape == (batch_size, model.num_classes)

    # # check logits
    # assert isinstance(logits, torch.Tensor)
    # assert all(isinstance(x.item(), float) for x in logits.flatten())

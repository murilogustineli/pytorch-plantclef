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

    # extract embeddings and logits from the predictions list
    all_embeddings = []
    all_logits = []
    for batch in dataloader:
        embed_batch, logits_batch = model.predict_step(
            batch, batch_idx=0
        )  # batch: List[Tuple[embeddings, logits]]
        all_embeddings.append(embed_batch)  # keep embeddings as tensors
        reshaped_logits = [
            logits_batch[i : i + grid_size**2]
            for i in range(0, len(logits_batch), grid_size**2)
        ]
        all_logits.extend(reshaped_logits)  # preserve batch structure

    # convert embeddings to tensor
    embeddings = torch.cat(all_embeddings, dim=0)  # shape: [len(df), grid_size**2, 768]
    logits = all_logits
    # logits = [logits.cpu().tolist() for logits in all_logits]

    if use_grid:
        embeddings = embeddings.view(-1, grid_size**2, 768)
    else:
        embeddings = embeddings.view(-1, 1, 768)

    assert isinstance(embeddings, torch.Tensor)
    assert all(isinstance(x.item(), float) for x in embeddings.flatten())

    expected_shape = (grid_size**2, expected_dim) if use_grid else (1, expected_dim)
    if use_grid:
        assert embeddings[0].shape == expected_shape
    else:
        assert embeddings[0].shape == expected_shape

    # check logits
    assert isinstance(logits, list)
    assert all(isinstance(inner_list, list) for inner_list in logits)
    assert all(isinstance(d, dict) for inner_list in logits for d in inner_list)
    assert all(
        len(d) == top_k for inner_list in logits for d in inner_list
    )  # ensure each dict has top-K entries
    assert all(
        isinstance(val, float)
        for inner_list in logits
        for d in inner_list
        for val in d.values()
    )  # ensure values are floats

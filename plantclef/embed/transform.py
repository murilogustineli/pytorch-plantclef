import timm
import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from plantclef.serde import deserialize_image
from plantclef.model_setup import setup_fine_tuned_model


class PlantDataset(Dataset):
    """Custom PyTorch Dataset for loading plant images from a Pandas DataFrame."""

    def __init__(self, df, transform=None):
        """
        Args:
            df (pd.DataFrame): Pandas DataFrame containing image binary data.
            transform (torchvision.transforms.Compose): Image transformations.
        """
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_bytes = self.df.iloc[idx]["data"]
        img = deserialize_image(img_bytes)  # convert from bytes to PIL image
        if self.transform:
            img = self.transform(img)
        return img


class DINOv2LightningModel(pl.LightningModule):
    """PyTorch Lightning module for extracting embeddings from a fine-tuned DINOv2 model."""

    def __init__(
        self,
        model_path=setup_fine_tuned_model(),
        model_name="vit_base_patch14_reg4_dinov2.lvd142m",
    ):
        super().__init__()
        self.model_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_classes = 7806  # total plant species

        # load the fine-tuned model
        self.model = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=self.num_classes,
            checkpoint_path=model_path,
        )

        # load transform
        self.data_config = timm.data.resolve_model_data_config(self.model)
        self.transform = timm.data.create_transform(
            **self.data_config, is_training=False
        )

        # move model to device
        self.model.to(self.model_device)
        self.model.eval()

    def forward(self, batch):
        """Extract embeddings using the [CLS] token."""
        with torch.no_grad():
            batch = batch.to(self.model_device)
            features = self.model.forward_features(batch)
            return features[:, 0, :]  # extract [CLS] token


def extract_embeddings(
    pandas_df: pd.DataFrame,
    batch_size: int = 32,
) -> np.ndarray:
    """Extract embeddings for images in a Pandas DataFrame using PyTorch Lightning."""

    # initialize model
    model = DINOv2LightningModel()

    # create Dataset and DataLoader
    dataset = PlantDataset(pandas_df, model.transform)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    # run inference and collect embeddings with tqdm progress bar
    all_embeddings = []
    for batch in tqdm(dataloader, desc="Extracting embeddings", unit="batch"):
        embeddings = model(batch)
        all_embeddings.append(embeddings.cpu().numpy())

    return np.vstack(all_embeddings)  # combine all embeddings into a single array

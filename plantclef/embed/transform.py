import io
import timm
import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from plantclef.model_setup import setup_fine_tuned_model


class PlantDataset(Dataset):
    """Custom PyTorch dataset for loading plant images from a Pandas DataFrame."""

    def __init__(self, df, transform):
        """
        Args:
            df (pd.DataFrame): Pandas DataFrame containing image binary data.
            transform (torchvision.transforms): Image transformations.
        """
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_bytes = self.df.iloc[idx]["image"]
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")  # Convert to RGB
        return self.transform(img)


class DINOv2LightningModel(pl.LightningModule):
    """PyTorch Lightning module for extracting embeddings from a fine-tuned DINOv2 model."""

    def __init__(
        self,
        model_path=setup_fine_tuned_model(),
        model_name="vit_base_patch14_reg4_dinov2.lvd142m",
    ):
        super().__init__()
        self.device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_classes = 7806  # Total plant species

        # Load the fine-tuned model
        self.model = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=self.num_classes,
            checkpoint_path=model_path,
        )

        # Load transform
        self.data_config = timm.data.resolve_model_data_config(self.model)
        self.transform = timm.data.create_transform(
            **self.data_config, is_training=False
        )

        # Move model to device
        self.model.to(self.device_type)
        self.model.eval()

    def forward(self, batch):
        """Extract embeddings using the [CLS] token."""
        with torch.no_grad():
            batch = batch.to(self.device_type)
            features = self.model.forward_features(batch)
            return features[:, 0, :]  # Extract CLS token


def extract_embeddings(
    train_df: pd.DataFrame,
    batch_size: int = 32,
) -> np.ndarray:
    """Extract embeddings for images in a Pandas DataFrame using PyTorch Lightning."""

    # Initialize model
    model = DINOv2LightningModel()

    # Create Dataset and DataLoader
    dataset = PlantDataset(train_df, model.transform)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    # Run inference and collect embeddings with tqdm progress bar
    all_embeddings = []
    for batch in tqdm(dataloader, desc="Extracting embeddings", unit="batch"):
        embeddings = model(batch)
        all_embeddings.append(embeddings.cpu().numpy())

    return np.vstack(all_embeddings)  # Combine all embeddings into a single array

import torch
import pytorch_lightning as pl

from functools import partial
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from plantclef.serde import deserialize_image
from plantclef.torch.model import DINOv2LightningModel


def custom_collate_fn(batch, use_grid):
    """Custom collate function to handle batched grid images properly."""
    if use_grid:
        return torch.stack(batch, dim=0)  # shape: (B, grid_size**2, C, H, W)
    return torch.stack(batch)  # shape: (B, C, H, W)


def custom_collate_fn_partial(use_grid):
    """Returns a pickle-friendly collate function with the `use_grid` flag."""
    return partial(custom_collate_fn, use_grid=use_grid)


class PlantDataset(Dataset):
    """Custom PyTorch Dataset for loading plant images from a Pandas DataFrame."""

    def __init__(
        self,
        df,
        transform=None,
        col_name: str = "data",
        use_grid: bool = False,
        grid_size: int = 4,
    ):
        """
        Args:
            df (pd.DataFrame): Pandas DataFrame containing image binary data.
            transform (torchvision.transforms.Compose): Image transformations.
            use_grid (bool): Whether to split images into a grid.
            grid_size (int): The size of the grid to split images into.
        """
        self.df = df
        self.transform = transform
        self.col_name = col_name
        self.use_grid = use_grid
        self.grid_size = grid_size

    def __len__(self):
        return len(self.df)

    def _split_into_grid(self, image):
        w, h = image.size
        grid_w, grid_h = w // self.grid_size, h // self.grid_size
        images = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                left = i * grid_w
                upper = j * grid_h
                right = left + grid_w
                lower = upper + grid_h
                crop_image = image.crop((left, upper, right, lower))
                images.append(crop_image)
        return images

    def __getitem__(self, idx) -> list:
        img_bytes = self.df.iloc[idx][self.col_name]  # column with image bytes
        img = deserialize_image(img_bytes)  # convert from bytes to PIL image

        if self.use_grid:
            img_list = self._split_into_grid(img)
            if self.transform:
                img_list = [self.transform(image) for image in img_list]
            else:  # no transform, shape: (grid_size**2, C, H, W)
                img_list = [ToTensor()(image) for image in img_list]
            return torch.stack(img_list)
        # single image, shape: (C, H, W)
        if self.transform:
            return self.transform(img)  # (C, H, W)
        return ToTensor()(img)  # (C, H, W)


class PlantDataModule(pl.LightningDataModule):
    """LightningDataModule for handling dataset loading and preparation."""

    def __init__(
        self, pandas_df, batch_size=32, use_grid=False, grid_size=4, num_workers=4
    ):
        super().__init__()
        self.pandas_df = pandas_df
        self.batch_size = batch_size
        self.use_grid = use_grid
        self.grid_size = grid_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        """Set up dataset and transformations."""

        self.model = DINOv2LightningModel()
        self.dataset = PlantDataset(
            self.pandas_df,
            self.model.transform,  # Use the model's transform
            use_grid=self.use_grid,
            grid_size=self.grid_size,
        )

    def predict_dataloader(self):
        """Returns DataLoader for inference."""
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=custom_collate_fn_partial(self.use_grid),
        )

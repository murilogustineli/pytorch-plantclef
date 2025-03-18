import timm
import torch
import pytorch_lightning as pl

from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from plantclef.serde import deserialize_image
from plantclef.model_setup import setup_fine_tuned_model
from plantclef.config import get_device, get_class_mappings_file


class PlantDataset(Dataset):
    """Custom PyTorch Dataset for loading plant images from a Pandas DataFrame."""

    def __init__(
        self,
        df,
        transform=None,
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
        img_bytes = self.df.iloc[idx]["data"]
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


class DINOv2LightningModel(pl.LightningModule):
    """PyTorch Lightning module for extracting embeddings from a fine-tuned DINOv2 model."""

    def __init__(
        self,
        model_path: str = setup_fine_tuned_model(),
        model_name: str = "vit_base_patch14_reg4_dinov2.lvd142m",
    ):
        super().__init__()
        self.model_device = get_device()
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
        # class mappings file for classification
        self.class_mappings_file = get_class_mappings_file()
        # load class mappings
        self.cid_to_spid = self._load_class_mappings()

    def _load_class_mappings(self):
        with open(self.class_mappings_file, "r") as f:
            class_index_to_class_name = {i: line.strip() for i, line in enumerate(f)}
        return class_index_to_class_name

    def forward(self, batch):
        """Extract embeddings using the [CLS] token."""
        with torch.no_grad():
            batch = batch.to(self.model_device)  # move to device

            if batch.dim() == 5:  # (B, grid_size**2, C, H, W)
                B, G, C, H, W = batch.shape
                batch = batch.view(B * G, C, H, W)  # (B * grid_size**2, C, H, W)
            # forward pass
            features = self.model.forward_features(batch)
            return features[:, 0, :]  # extract [CLS] token

import timm
import torch
import pytorch_lightning as pl

from plantclef.model_setup import setup_fine_tuned_model
from plantclef.config import get_device, get_class_mappings_file


class DINOv2LightningModel(pl.LightningModule):
    """PyTorch Lightning module for extracting embeddings from a fine-tuned DINOv2 model."""

    def __init__(
        self,
        model_path: str = setup_fine_tuned_model(),
        model_name: str = "vit_base_patch14_reg4_dinov2.lvd142m",
        top_k: int = 10,
    ):
        super().__init__()
        self.model_device = get_device()
        self.num_classes = 7806  # total plant species
        self.top_k = top_k

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
            embeddings = features[:, 0, :]  # extract [CLS] token
            logits = self.model(batch)

        return embeddings, logits

    def predict_step(self, batch, batch_idx):
        """Runs inference on batch and returns embeddings and top-K logits."""
        embeddings, logits = self(batch)
        probabilities = torch.softmax(logits, dim=1)
        top_probs, top_indices = torch.topk(probabilities, k=self.top_k, dim=1)

        # map class indices to species names
        batch_logits = []
        for i in range(len(logits)):
            species_probs = {
                self.cid_to_spid.get(int(top_indices[i, j].item()), "Unknown"): float(
                    top_probs[i, j].item()
                )
                for j in range(self.top_k)
            }
            batch_logits.append(species_probs)

        return embeddings, batch_logits

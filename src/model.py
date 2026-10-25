from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import roc_auc_score, roc_curve

class ResNetAnomalyDetector:
    """
    Budget-aware Anomaly Detector using pretrained ResNet-18 feature embeddings.
    Extracts high-level semantic features, learns the 'Center of Normalcy' from
    defect-free training data, and scores queries via distance to the normal distribution.
    """
    def __init__(self, device=None, backbone="resnet18", scoring_method="centroid"):
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.scoring_method = scoring_method  # 'centroid' or 'knn'
        self.model = self._build_backbone(backbone)
        self.normal_embeddings = None
        self.normal_centroid = None
        self.threshold = None

    def _build_backbone(self, backbone: str) -> nn.Module:
        if backbone == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            net = models.resnet18(weights=weights)
        elif backbone == "resnet34":
            weights = models.ResNet34_Weights.IMAGENET1K_V1
            net = models.resnet34(weights=weights)
        elif backbone == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V1
            net = models.resnet50(weights=weights)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Replace classification head with Identity to extract raw pooled embeddings
        net.fc = nn.Identity()
        net.eval()
        for p in net.parameters():
            p.requires_grad = False
        return net.to(self.device)

    def extract_features(self, dataloader: DataLoader) -> tuple[np.ndarray, np.ndarray, list[dict]]:
        """Extracts deep features from a dataloader without storing computational graphs."""
        features_list = []
        labels_list = []
        metadata_list = []

        with torch.no_grad():
            for batch in dataloader:
                imgs = batch["image"].to(self.device)
                feats = self.model(imgs)
                features_list.append(feats.cpu().numpy())
                labels_list.append(batch["label"].numpy())

                batch_size = len(batch["label"])
                for i in range(batch_size):
                    metadata_list.append({
                        "filename": batch["filename"][i],
                        "defect_type": batch["defect_type"][i],
                        "image_path": batch["image_path"][i],
                        "mask_path": batch["mask_path"][i]
                    })

        features = np.concatenate(features_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)
        return features, labels, metadata_list

    def fit(self, train_loader: DataLoader):
        """Fit on normal training samples by computing the normal distribution archetype."""
        train_features, _, _ = self.extract_features(train_loader)
        self.normal_embeddings = train_features
        self.normal_centroid = np.mean(train_features, axis=0)
        
        # Internal calibration: compute train self-scores to calibrate nominal distribution
        train_distances = np.linalg.norm(train_features - self.normal_centroid, axis=1)
        # Default decision threshold set to 95th percentile of normal training spread
        self.threshold = float(np.percentile(train_distances, 95))
        return self

    def score_features(self, features: np.ndarray) -> np.ndarray:
        """Compute anomaly scores from extracted feature vectors."""
        if self.normal_centroid is None:
            raise RuntimeError("Model must be fitted on normal training data before scoring.")

        if self.scoring_method == "centroid":
            scores = np.linalg.norm(features - self.normal_centroid, axis=1)
        elif self.scoring_method == "knn":
            # k-NN distance (k=3)
            from sklearn.neighbors import NearestNeighbors
            nn_model = NearestNeighbors(n_neighbors=min(3, len(self.normal_embeddings)), metric="euclidean")
            nn_model.fit(self.normal_embeddings)
            distances, _ = nn_model.kneighbors(features)
            scores = np.mean(distances, axis=1)
        else:
            raise ValueError(f"Unknown scoring method: {self.scoring_method}")
            
        return scores

    def score_single_image(self, pil_image, transform) -> float:
        """Inference for a single PIL image in Streamlit or production API."""
        img_tensor = transform(pil_image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model(img_tensor).cpu().numpy()
        score = float(self.score_features(feat)[0])
        return score

    def evaluate(self, test_loader: DataLoader) -> dict:
        """Evaluates image-level detection metrics on test split."""
        test_features, test_labels, test_meta = self.extract_features(test_loader)
        test_scores = self.score_features(test_features)

        auroc = float(roc_auc_score(test_labels, test_scores))
        fpr, tpr, thresholds = roc_curve(test_labels, test_scores)

        # Flagged by nominal threshold
        predicted_defect = (test_scores >= self.threshold).astype(int)

        return {
            "auroc": auroc,
            "fpr": fpr,
            "tpr": tpr,
            "threshold": self.threshold,
            "scores": test_scores,
            "labels": test_labels,
            "metadata": test_meta,
            "predicted_defect": predicted_defect
        }

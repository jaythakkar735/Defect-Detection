from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

def get_default_transforms(img_size=(256, 256)):
    """Standard transforms for ImageNet-pretrained CNN backbones."""
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

class MVTecCategoryDataset(Dataset):
    """
    Modular MVTec AD Dataset for any object or texture category.
    Labels: 0 = Normal / Good, 1 = Defective / Anomalous.
    """
    def __init__(self, category_dir: Path, split: str = "train", transform=None):
        self.category_dir = Path(category_dir)
        self.split = split
        self.transform = transform if transform is not None else get_default_transforms()
        
        self.samples = []
        self._load_samples()

    def _load_samples(self):
        split_dir = self.category_dir / self.split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory does not exist: {split_dir}")

        if self.split == "train":
            # Normal training images only
            good_dir = split_dir / "good"
            for img_path in sorted(good_dir.glob("*.png")):
                self.samples.append({
                    "image_path": img_path,
                    "filename": img_path.name,
                    "label": 0,
                    "defect_type": "good",
                    "mask_path": None
                })
        elif self.split == "test":
            # Test split contains both 'good' and multiple defect subdirectories
            for defect_dir in sorted(split_dir.iterdir()):
                if not defect_dir.is_dir():
                    continue
                defect_name = defect_dir.name
                is_defect = defect_name != "good"
                label = 1 if is_defect else 0
                
                for img_path in sorted(defect_dir.glob("*.png")):
                    mask_path = None
                    if is_defect:
                        expected_mask = self.category_dir / "ground_truth" / defect_name / f"{img_path.stem}_mask.png"
                        if expected_mask.exists():
                            mask_path = expected_mask
                    
                    self.samples.append({
                        "image_path": img_path,
                        "filename": img_path.name,
                        "label": label,
                        "defect_type": defect_name,
                        "mask_path": mask_path
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        with Image.open(item["image_path"]) as img:
            img = img.convert("RGB")
            img_tensor = self.transform(img)

        return {
            "image": img_tensor,
            "label": item["label"],
            "defect_type": item["defect_type"],
            "image_path": str(item["image_path"]),
            "filename": item["image_path"].name,
            "mask_path": str(item["mask_path"]) if item["mask_path"] else ""
        }

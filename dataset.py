
import torch
from PIL import Image, ImageFile, UnidentifiedImageError
from torch.utils.data import Dataset
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True


class MyDataset(Dataset):
    def __init__(self, dataframe, training=False, horizontal_flip=False):
        self.data = dataframe.reset_index(drop=True).copy()

        required_columns = {"image_path", "label"}
        missing = required_columns - set(self.data.columns)

        if missing:
            raise ValueError(f"Missing dataset columns: {missing}")

        self.training = training

        # Image preprocessing
        steps = [
            transforms.Resize((224, 224))
        ]

        # Optional data augmentation for training only
        if training and horizontal_flip:
            steps.append(
                transforms.RandomHorizontalFlip(p=0.5)
            )

        # Convert image to tensor and normalize
        steps.extend([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.transform = transforms.Compose(steps)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        image_path = str(row["image_path"])
        label = int(row["label"])

        # Validate binary classification labels
        if label not in (0, 1):
            raise ValueError(
                f"Invalid label {label} for image: {image_path}"
            )

        try:
            # Safely load image and convert to RGB
            with Image.open(image_path) as img:
                image = img.convert("RGB")

        except (OSError, ValueError, UnidentifiedImageError) as e:
            raise RuntimeError(
                f"Failed to load X-ray image: {image_path}"
            ) from e

        # Apply preprocessing
        image = self.transform(image)

        # Convert label to PyTorch tensor
        target = torch.tensor(label, dtype=torch.long)

        return {
            "xray": image,
            "target": target
        }

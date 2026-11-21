
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader

from args import get_args
from data_preparation import make_csvs
from dataset import MyDataset
from model import MyModel
from trainer import train_model
from evaluate import evaluate_model


def main():
    # Read command-line arguments
    args = get_args()

    # Validate training settings
    if args.epochs < 1 or args.bs < 1:
        raise ValueError(
            "Epochs and batch size must be positive."
        )

    # Prepare CSV files and remove exact duplicates
    make_csvs(args.data_dir, args.output_dir)

    # Load training, validation, and test CSV files
    train_df = pd.read_csv(
        os.path.join(args.output_dir, "training.csv")
    )

    val_df = pd.read_csv(
        os.path.join(args.output_dir, "validation.csv")
    )

    test_df = pd.read_csv(
        os.path.join(args.output_dir, "test.csv")
    )

    # Create datasets
    train_dataset = MyDataset(
        train_df,
        training=True,
        horizontal_flip=args.horizontal_flip
    )

    val_dataset = MyDataset(
        val_df,
        training=False
    )

    test_dataset = MyDataset(
        test_df,
        training=False
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.bs,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.bs,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.bs,
        shuffle=False
    )

    # Select CPU or GPU
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)
    print("Backbone:", args.backbone)
    print("Learning rate:", args.lr)
    print("Batch size:", args.bs)
    print("Epochs:", args.epochs)
    print("Horizontal flip:", args.horizontal_flip)

    # Initialize pretrained ResNet model
    model = MyModel(args.backbone).to(device)

    # Train and save best validation model
    model_path = train_model(
        model,
        train_loader,
        val_loader,
        args,
        device
    )

    # Load best saved model
    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device,
            weights_only=True
        )
    )

    # Evaluate on test dataset
    print("\nTest results:")

    evaluate_model(
        model,
        test_loader,
        device,
        args.output_dir
    )


if __name__ == "__main__":
    main()

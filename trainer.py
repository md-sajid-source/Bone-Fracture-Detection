
import os
import torch
import matplotlib.pyplot as plt
from torch import nn, optim
from sklearn.metrics import accuracy_score


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)

    total_loss = 0
    actual, predicted = [], []

    for batch in loader:
        images = batch["xray"].to(device)
        labels = batch["target"].to(device)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            outputs = model(images)
            loss = criterion(outputs, labels)

            if training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * labels.size(0)
        actual.extend(labels.cpu().tolist())
        predicted.extend(
            outputs.argmax(1).detach().cpu().tolist()
        )

    avg_loss = total_loss / len(loader.dataset)
    accuracy = accuracy_score(actual, predicted)

    return avg_loss, accuracy


def train_model(model, train_loader, val_loader, args, device):
    os.makedirs(args.output_dir, exist_ok=True)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.wd
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2
    )

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    best_accuracy = -1.0
    best_loss = float("inf")
    patience = 5
    epochs_without_improvement = 0

    model_path = os.path.join(
        args.output_dir, "best_model.pth"
    )

    for epoch in range(args.epochs):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, device, optimizer
        )

        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, device
        )

        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"train loss: {train_loss:.4f} | "
            f"train accuracy: {train_acc:.4f} | "
            f"val loss: {val_loss:.4f} | "
            f"val accuracy: {val_acc:.4f} | "
            f"lr: {current_lr:.6f}"
        )

        # Save the best model using validation accuracy
        # and validation loss as a tiebreaker.
        improved = (
            val_acc > best_accuracy + 1e-8
            or (
                abs(val_acc - best_accuracy) <= 1e-8
                and val_loss < best_loss
            )
        )

        if improved:
            best_accuracy = val_acc
            best_loss = val_loss
            epochs_without_improvement = 0

            torch.save(model.state_dict(), model_path)
            print("Best model saved.")
        else:
            epochs_without_improvement += 1

        # Stop if validation performance does not improve
        if epochs_without_improvement >= patience:
            print("Early stopping triggered.")
            break

    # Plot training and validation loss
    plt.figure()
    plt.plot(train_losses, label="Training")
    plt.plot(val_losses, label="Validation")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "loss.png"))
    plt.close()

    # Plot training and validation accuracy
    plt.figure()
    plt.plot(train_accs, label="Training")
    plt.plot(val_accs, label="Validation")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "accuracy.png"))
    plt.close()

    print(f"Best validation accuracy: {best_accuracy:.4f}")
    print(f"Checkpoint: {model_path}")

    # IMPORTANT: Return the checkpoint path to main.py
    return model_path

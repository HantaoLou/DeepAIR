"""Training script for antibody-antigen complex modality."""

import argparse
import math
import os
import sys

# Allow imports when executed from the ``maincode`` directory.
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from deepair.datasets.antibody_antigen import (
    load_antibody_antigen_dataset,
    batch_generator,
)


def _mean_feature(seqs, max_len):
    """Compute the mean of encoded sequences up to ``max_len`` residues."""
    features = []
    for seq in seqs:
        if not seq:
            features.append(0.0)
            continue
        truncated = seq[:max_len]
        features.append(sum(truncated) / float(len(truncated)))
    return features


def train(train_csv, epochs=10, lr=0.1, batch_size=32, max_len=32, model_path="antibody_antigen_model.txt"):
    """Train a tiny logistic regression model.

    Parameters
    ----------
    train_csv : str
        Path to the training CSV file.
    epochs : int
        Number of training epochs.
    lr : float
        Learning rate for gradient descent.
    batch_size : int
        Mini-batch size.
    max_len : int
        Maximum sequence length considered when computing features.
    model_path : str
        File to store the trained model weights.
    """
    antibodies, antigens, labels = load_antibody_antigen_dataset(train_csv)
    w = [0.0, 0.0]
    b = 0.0
    for _ in range(epochs):
        for ab_batch, ag_batch, lb_batch in batch_generator(antibodies, antigens, labels, batch_size=batch_size):
            ab_feat = _mean_feature(ab_batch, max_len)
            ag_feat = _mean_feature(ag_batch, max_len)
            for af, gf, y in zip(ab_feat, ag_feat, lb_batch):
                z = w[0] * af + w[1] * gf + b
                pred = 1.0 / (1.0 + math.exp(-z))
                error = pred - y
                w[0] -= lr * error * af
                w[1] -= lr * error * gf
                b -= lr * error
    with open(model_path, "w") as f:
        f.write(f"{w[0]},{w[1]},{b}")
    return w, b


def load_model(model_path):
    """Load model weights from ``model_path``."""
    with open(model_path) as f:
        w0, w1, b = map(float, f.read().strip().split(","))
    return [w0, w1], b


def evaluate(model_path, test_csv, batch_size=32, max_len=32):
    """Evaluate a saved model on a test CSV file."""
    w, b = load_model(model_path)
    antibodies, antigens, labels = load_antibody_antigen_dataset(test_csv)
    total, correct = 0, 0
    for ab_batch, ag_batch, lb_batch in batch_generator(
        antibodies, antigens, labels, batch_size=batch_size, shuffle=False
    ):
        ab_feat = _mean_feature(ab_batch, max_len)
        ag_feat = _mean_feature(ag_batch, max_len)
        for af, gf, y in zip(ab_feat, ag_feat, lb_batch):
            z = w[0] * af + w[1] * gf + b
            pred = 1.0 / (1.0 + math.exp(-z))
            pred_label = 1 if pred >= 0.5 else 0
            if pred_label == int(y):
                correct += 1
            total += 1
    acc = correct / total if total else 0.0
    print(f"Test accuracy: {acc:.4f}")
    return acc


def main():
    parser = argparse.ArgumentParser(description="Train antibody-antigen complex model")
    parser.add_argument("--train_csv", required=True, help="Training CSV file")
    parser.add_argument("--model_path", default="antibody_antigen_model.txt", help="Where to store the trained model")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_len", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.1)
    args = parser.parse_args()
    train(
        args.train_csv,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        max_len=args.max_len,
        model_path=args.model_path,
    )


if __name__ == "__main__":
    main()

"""Evaluate a trained antibody-antigen model on a test set."""

import argparse

from train_antibody_antigen import evaluate


def main():
    parser = argparse.ArgumentParser(description="Test antibody-antigen complex model")
    parser.add_argument("--model_path", required=True, help="Path to trained model file")
    parser.add_argument("--test_csv", required=True, help="CSV file containing test data")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_len", type=int, default=32)
    args = parser.parse_args()
    evaluate(args.model_path, args.test_csv, batch_size=args.batch_size, max_len=args.max_len)


if __name__ == "__main__":
    main()

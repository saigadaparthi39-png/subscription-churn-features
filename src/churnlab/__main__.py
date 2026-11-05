"""Generate data, train the baseline, or write batch scores."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from churnlab.generate import generate
from churnlab.train import score, train


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="churnlab")
    sub = parser.add_subparsers(dest="cmd", required=True)

    gen = sub.add_parser("generate", help="write synthetic account history")
    gen.add_argument("--out", default="data/raw")
    gen.add_argument("--accounts", type=int, default=2000)
    gen.add_argument("--seed", type=int, default=7)

    fit = sub.add_parser("train", help="fit the logistic baseline and write metrics")
    fit.add_argument("--raw", default="data/raw")
    fit.add_argument("--model", default="data/model.joblib")
    fit.add_argument("--metrics", default="data/metrics.json")

    batch = sub.add_parser("score", help="score accounts that are still active")
    batch.add_argument("--raw", default="data/raw")
    batch.add_argument("--model", default="data/model.joblib")
    batch.add_argument("--out", default="data/scores.csv")

    args = parser.parse_args(argv)
    if args.cmd == "generate":
        generate(Path(args.out), n_accounts=args.accounts, seed=args.seed)
        print(f"wrote raw files under {args.out}")
        return 0
    if args.cmd == "train":
        metrics = train(Path(args.raw), Path(args.model), Path(args.metrics))
        print(json.dumps({k: v for k, v in metrics.items() if k != "features"}, indent=2))
        return 0
    written = score(Path(args.raw), Path(args.model), Path(args.out))
    print(f"scored {written} accounts -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

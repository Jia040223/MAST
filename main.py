import argparse
import importlib.util
from pathlib import Path

import run_lib


def load_config(config_path: str):
    spec = importlib.util.spec_from_file_location("mast_config", config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_config()


def parse_args():
    parser = argparse.ArgumentParser(description="MAST training and diffusion sampling")
    parser.add_argument("--config", required=True, help="Path to a config file.")
    parser.add_argument("--workdir", required=True, help="Output directory.")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["train", "sample"],
        help="Run training or diffusion sampling.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint path for sampling. If omitted, the latest configured checkpoint is used.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output pickle for sampling results. Defaults to <workdir>/samples/generated.pkl.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    if args.mode == "train":
        run_lib.train(config, str(workdir))
    else:
        output_path = args.output or str(workdir / "samples" / "generated.pkl")
        run_lib.sample(config, str(workdir), checkpoint_path=args.checkpoint, output_path=output_path)


if __name__ == "__main__":
    main()

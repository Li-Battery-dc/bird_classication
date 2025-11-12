import argparse
from pathlib import Path

from svm.Method import svm_method
from cnn.Method import train as cnn_train
from cnn.Method import validate as cnn_validate
from cnn.visulize_utils import visualizes_all
from ViT.train import train_main as vit_train
from ViT.visualize import plot_training_curves, visualize_attention


def existing_path(path_str: str) -> str:
    """Validates that the given path exists before continuing."""
    path = Path(path_str).expanduser()
    if not path.exists():
        raise argparse.ArgumentTypeError(f"Path does not exist: {path}")
    return str(path)


def ensure_parent_dir(path_str: str) -> str:
    """Ensures the parent directory exists for paths we write to."""
    path = Path(path_str).expanduser()
    if path.parent and not path.parent.exists():
        raise argparse.ArgumentTypeError(
            f"Parent directory does not exist for: {path}"
        )
    return str(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bird classification utilities CLI"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # CNN training
    train_parser = subparsers.add_parser(
        "cnn-train", help="Train the CNN classifier"
    )
    train_parser.add_argument(
        "--weight-save-path",
        type=ensure_parent_dir,
        required=True,
        help="Destination path for the final model weights (.pth)",
    )
    train_parser.add_argument(
        "--ckpt-load-path",
        type=existing_path,
        default=None,
        help="Optional checkpoint path to resume training",
    )
    train_parser.add_argument(
        "--data-root",
        default="./data/",
        help="Dataset root directory",
    )
    train_parser.add_argument(
        "--freeze",
        dest="enable_freeze",
        action="store_true",
        default=True,
        help="Enable backbone freezing (default)",
    )
    train_parser.add_argument(
        "--no-freeze",
        dest="enable_freeze",
        action="store_false",
        help="Disable backbone freezing",
    )
    train_parser.add_argument(
        "--freeze-epoch-ratio",
        type=float,
        default=0.75,
        help="Ratio of epochs to keep layers frozen when enabled",
    )
    train_parser.add_argument(
        "--warmup",
        dest="enable_warmup",
        action="store_true",
        default=True,
        help="Enable warmup scheduler (default)",
    )
    train_parser.add_argument(
        "--no-warmup",
        dest="enable_warmup",
        action="store_false",
        help="Disable warmup scheduler",
    )
    train_parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Warmup ratio when warmup is enabled",
    )
    train_parser.add_argument(
        "--num-classes",
        type=int,
        default=200,
        help="Number of target classes",
    )
    train_parser.add_argument(
        "--num-epochs",
        type=int,
        default=1000,
        help="Number of training epochs",
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Training batch size",
    )

    # CNN validation
    validate_parser = subparsers.add_parser(
        "cnn-validate", help="Validate a trained CNN checkpoint"
    )
    validate_parser.add_argument(
        "--weight-path",
        type=existing_path,
        required=True,
        help="Path to the trained weights (.pth)",
    )
    validate_parser.add_argument(
        "--data-root",
        default="./data/",
        help="Dataset root directory",
    )
    validate_parser.add_argument(
        "--num-classes",
        type=int,
        default=200,
        help="Number of target classes",
    )

    # CNN visualization
    visualize_parser = subparsers.add_parser(
        "cnn-visualize", help="Visualize training and checkpoint metrics"
    )
    visualize_parser.add_argument(
        "--log-file",
        type=existing_path,
        required=True,
        help="Training log file to parse",
    )
    visualize_parser.add_argument(
        "--ckpts-dir",
        type=existing_path,
        required=True,
        help="Directory containing checkpoint files",
    )
    visualize_parser.add_argument(
        "--result-dir",
        default="./result/cnn",
        help="Base directory for visualization outputs",
    )
    visualize_parser.add_argument(
        "--data-root",
        default="./data/",
        help="Dataset root directory",
    )

    # SVM baseline
    svm_parser = subparsers.add_parser(
        "svm", help="Run the SVM baseline experiment"
    )
    svm_parser.add_argument(
        "--data-root",
        default="./data",
        help="Dataset root directory",
    )
    svm_parser.add_argument(
        "--kernel",
        choices=["linear", "rbf", "poly"],
        default="linear",
        help="SVM kernel type",
    )
    svm_parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Regularization parameter C",
    )
    svm_parser.add_argument(
        "--gamma",
        default="scale",
        help="Gamma for RBF/poly ('scale' or numeric value)",
    )
    svm_parser.add_argument(
        "--degree",
        type=int,
        default=3,
        help="Degree for polynomial kernel",
    )
    svm_parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-5,
        help="Threshold for support vector filtering (numerical stability)",
    )
    svm_parser.add_argument(
        "--num-classes",
        type=int,
        default=10,
        help="Number of classes to randomly sample for the experiment",
    )
    svm_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for class sampling",
    )

    # ViT training
    vit_train_parser = subparsers.add_parser(
        "vit-train", help="Train Vision Transformer (ViT) model"
    )
    vit_train_parser.add_argument(
        "--config-note",
        type=str,
        default="",
        help="配置说明：请直接修改 src/ViT/config.py 文件来调整训练参数"
    )

    # ViT visualization
    vit_vis_parser = subparsers.add_parser(
        "vit-visualize", help="Visualize ViT training curves or attention maps"
    )
    vit_vis_subparsers = vit_vis_parser.add_subparsers(
        dest="vis_mode", required=True,
        help="Visualization mode: curves or attention"
    )

    # ViT curves visualization
    vit_curves_parser = vit_vis_subparsers.add_parser(
        "curves", help="Visualize training and validation curves from log"
    )
    vit_curves_parser.add_argument(
        "--log",
        type=existing_path,
        required=True,
        help="Training log file path (e.g., result/vit/train_xxx/train.log)"
    )
    vit_curves_parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory to save visualization (default: same as log file)"
    )

    # ViT attention visualization
    vit_attn_parser = vit_vis_subparsers.add_parser(
        "attention", help="Visualize attention maps for a specific image"
    )
    vit_attn_parser.add_argument(
        "--image",
        type=existing_path,
        required=True,
        help="Input image path"
    )
    vit_attn_parser.add_argument(
        "--model",
        type=existing_path,
        required=True,
        help="Model checkpoint path (e.g., result/vit/train_xxx/ckpt/best_model.pth)"
    )
    vit_attn_parser.add_argument(
        "--layer",
        type=int,
        default=-1,
        help="Transformer layer index (-1 for last layer, 0-11 for specific layer)"
    )
    vit_attn_parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory to save visualization (default: result/vit/vis_images)"
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "cnn-train":
        cnn_train(
            weight_save_path=args.weight_save_path,
            ckpt_load_path=args.ckpt_load_path,
            data_root=args.data_root,
            enable_freeze=args.enable_freeze,
            freeze_epoch_ratio=args.freeze_epoch_ratio,
            enable_warmup=args.enable_warmup,
            warmup_ratio=args.warmup_ratio,
            num_classes=args.num_classes,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
        )
    elif args.command == "cnn-validate":
        cnn_validate(
            weight_path=args.weight_path,
            data_root=args.data_root,
            num_classes=args.num_classes,
        )
    elif args.command == "cnn-visualize":
        visualizes_all(
            log_file=args.log_file,
            ckpts_dir=args.ckpts_dir,
            result_dir=args.result_dir,
            data_root=args.data_root,
        )
    elif args.command == "svm":
        svm_method(
            data_root=args.data_root,
            kernel=args.kernel,
            C=args.C,
            gamma=args.gamma,
            degree=args.degree,
            epsilon=args.epsilon,
            num_classes=args.num_classes,
            seed=args.seed,
        )
    elif args.command == "vit-train":
        print("\n" + "=" * 60)
        print("Vision Transformer (ViT) Training")
        print("=" * 60)
        print("⚙️  配置说明: 请直接修改 src/ViT/config.py 文件来调整训练参数")
        print("=" * 60 + "\n")
        vit_train()
    elif args.command == "vit-visualize":
        if args.vis_mode == "curves":
            plot_training_curves(args.log, args.save_dir)
        elif args.vis_mode == "attention":
            visualize_attention(
                image_path=args.image,
                model_path=args.model,
                layer_idx=args.layer,
                save_dir=args.save_dir
            )
    else:
        parser.error("Unknown command")


if __name__ == "__main__":
    main()
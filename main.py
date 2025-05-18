import argparse
from pipeLinePerFamilyToAll import main as pipeline_per_family_to_all
from perfamilyPipLine import main as pipeline_per_family
from PipLineAllData import main as pipeline_all_data
from pipeLine_supervisedEcappa import main as pipeline_supervised
from Ecappa_training_loop_superviesd import main as Ecappa_pipeline_supervised
from Ecappa_training_loop_unSupervised import main as Ecappa_pipeline_unsupervised

from model_funcs import *
from utils import *
import wandb

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run different pipelines for speaker recognition.")
    parser.add_argument(
        "--pipeline",
        type=str,
        required=True,
        choices=["per_family_to_all", "per_family", "supervised", "all_data", "ecappa_supervised", "ecappa_unsupervised"],
        help="Specify which pipeline to run: 'per_family_to_all', 'per_family', 'supervised', 'ecappa_supervised', 'ecappa_unsupervised',  or 'all_data'."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="/app/config.yaml",
        help="Path to the configuration file (default: /app/config.yaml)."
    )
    args = parser.parse_args()

    # Load configuration settings
    config = load_config(args.config)

    # Initialize wandb
    run = wandb.init(
        entity="oribaruch-engineering",
        project="graduation_project",
    )

    # Run the selected pipeline
    if args.pipeline == "per_family_to_all":
        print("Running pipeline: Per Family to All")
        pipeline_per_family_to_all(run, args.config)
    elif args.pipeline == "per_family":
        print("Running pipeline: Per Family")
        pipeline_per_family(run, args.config)
    elif args.pipeline == "supervised":
        print("Running pipeline: supervised")
        pipeline_supervised(run, args.config)
    elif args.pipeline == "all_data":
        print("Running pipeline: All Data")
        pipeline_all_data(run, args.config)
    elif args.pipeline == "ecappa_supervised":
        print("Running pipeline: All Data ecappa_supervised")
        Ecappa_pipeline_supervised(run, args.config)
    elif args.pipeline == "ecappa_unsupervised":
        print("Running pipeline: All Data ecappa_unsupervised")
        Ecappa_pipeline_unsupervised(run, args.config)
    else:
        print("Invalid pipeline selected. Use --help for usage information.")

    # Finish the wandb run
    run.finish()


if __name__ == "__main__":
    main()

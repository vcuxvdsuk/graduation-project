import argparse

from triplet_training_loop_supervised import main as triplet_pipeline_supervised
from triplet_training_loop_unSupervised import main as triplet_pipeline_unsupervised
from triplet_training_loop_unSupervised_family import main as triplet_pipeline_per_family_to_all

from Ecappa_training_loop_supervised import main as Ecappa_pipeline_supervised
from Ecappa_training_loop_unSupervised import main as Ecappa_pipeline_unsupervised
from Ecappa_training_loop_unSupervised_family import main as Ecappa_pipeline_unsupervised_family

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
        choices=[ "triplet_supervised", "triplet_unsupervised", "triplet_unsupervised_family",
                  "ecappa_supervised", "ecappa_unsupervised", "ecappa_unsupervised_family"],
        help="Specify which pipeline to run: 'triplet_supervised', 'triplet_unsuperised', 'triplet_per_family', 'ecappa_supervised', 'ecappa_unsupervised', 'ecappa_unsupervised_family'. '"
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
    if args.pipeline == "triplet_unsupervised_family":
        print("Running pipeline: Per Family to All")
        triplet_pipeline_per_family_to_all(run, args.config)
    elif args.pipeline == "triplet_supervised":
        print("Running pipeline: supervised")
        triplet_pipeline_supervised(run, args.config)
    elif args.pipeline == "triplet_unsupervised":
        print("Running pipeline: All Data")
        triplet_pipeline_unsupervised(run, args.config)
    elif args.pipeline == "ecappa_supervised":
        print("Running pipeline: All Data ecappa_supervised")
        Ecappa_pipeline_supervised(run, args.config)        
    elif args.pipeline == "ecappa_unsupervised":
        print("Running pipeline: All Data ecappa_unsupervised")
        Ecappa_pipeline_unsupervised(run, args.config)
    elif args.pipeline == "ecappa_unsupervised_family":
        print("Running pipeline: All Data ecappa_unsupervised_family")
        Ecappa_pipeline_unsupervised_family(run, args.config)
    else:
        print("Invalid pipeline selected. Use --help for usage information.")

    # Finish the wandb run
    run.finish()


if __name__ == "__main__":
    main()

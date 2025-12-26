import torch
import numpy as np
from torch.utils.data import DataLoader, Subset

from data.cifar10 import get_cifar10
from models.resnet import get_model
from al.strategies import random_sampling, entropy_sampling
from train import train_one_epoch, evaluate
from utils import load_config, init_logger, log_row


# --------------------------------------------------
# Setup
# --------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CFG = load_config("config.yaml")


# --------------------------------------------------
# Core experiment function
# --------------------------------------------------
def run_experiment(strategy_name: str):
    """
    Run one active learning experiment.

    strategy_name: "random" or "entropy"
    """

    assert strategy_name in ["random", "entropy"], "Invalid strategy name"

    print(f"\n========== Running {strategy_name.upper()} Sampling ==========")

    # Load dataset
    train_set, test_set = get_cifar10()
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False)

    # Initialize labeled / unlabeled pools
    labeled_idx = np.random.choice(
        len(train_set),
        CFG["init_labeled"],
        replace=False
    )
    unlabeled_idx = np.setdiff1d(np.arange(len(train_set)), labeled_idx)

    # Initialize model & optimizer
    model = get_model().to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CFG["learning_rate"]
    )

    # Initialize logger
    log_path = f"results/{strategy_name}.csv"
    init_logger(log_path)

    # --------------------------------------------------
    # Active Learning Loop
    # --------------------------------------------------
    for round_id in range(CFG["rounds"]):
        print(f"\n[{strategy_name.upper()}] Round {round_id}")

        # Training loader
        train_loader = DataLoader(
            Subset(train_set, labeled_idx),
            batch_size=CFG["batch_size"],
            shuffle=True
        )

        # Train model
        for _ in range(CFG["epochs_per_round"]):
            train_one_epoch(model, train_loader, optimizer, DEVICE)

        # Evaluate
        acc = evaluate(model, test_loader, DEVICE)
        print(f"Test Accuracy: {acc:.4f}")

        # Log results
        log_row(
            log_path,
            round_id,
            len(labeled_idx),
            acc
        )

        # Stop if last round
        if round_id == CFG["rounds"] - 1:
            break

        # --------------------------------------------------
        # Query new samples
        # --------------------------------------------------
        if strategy_name == "random":
            query_idx = random_sampling(
                unlabeled_idx,
                CFG["query_size"]
            )

        else:  # entropy sampling
            query_loader = DataLoader(
                Subset(train_set, unlabeled_idx),
                batch_size=128,
                shuffle=False
            )

            query_idx = entropy_sampling(
                model=model,
                dataloader=query_loader,
                unlabeled_idx=unlabeled_idx,
                n=CFG["query_size"],
                device=DEVICE
            )

        # Update pools
        labeled_idx = np.concatenate([labeled_idx, query_idx])
        unlabeled_idx = np.setdiff1d(unlabeled_idx, query_idx)


# --------------------------------------------------
# Main
# --------------------------------------------------
if __name__ == "__main__":
    run_experiment("random")
    run_experiment("entropy")
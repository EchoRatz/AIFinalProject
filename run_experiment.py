import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from data.cifar10 import get_cifar10
from models.resnet import get_model
from al.strategies import random_sampling, entropy_sampling
from train import train_one_epoch, evaluate

device = "cuda" if torch.cuda.is_available() else "cpu"

train_set, test_set = get_cifar10()
test_loader = DataLoader(test_set, batch_size=256)

labeled_idx = np.random.choice(len(train_set), 1000, replace=False)
unlabeled_idx = np.setdiff1d(np.arange(len(train_set)), labeled_idx)

model = get_model().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for round in range(10):
    print(f"\n=== Round {round} ===")

    train_loader = DataLoader(
        Subset(train_set, labeled_idx),
        batch_size=64,
        shuffle=True
    )

    for _ in range(5):
        train_one_epoch(model, train_loader, optimizer, device)

    acc = evaluate(model, test_loader, device)
    print(f"Test accuracy: {acc:.4f}")

    query = random_sampling(unlabeled_idx, 1000)
    labeled_idx = np.concatenate([labeled_idx, query])
    unlabeled_idx = np.setdiff1d(unlabeled_idx, query)

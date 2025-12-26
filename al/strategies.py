import torch
import torch.nn.functional as F
import numpy as np


def random_sampling(unlabeled_idx, n):
    return np.random.choice(unlabeled_idx, n, replace=False)


def entropy_sampling(model, dataloader, unlabeled_idx, n, device):
    """
    Robust entropy-based sampling.
    Does NOT rely on dataset returning indices.
    """
    model.eval()
    entropies = []

    with torch.no_grad():
        for images, _, in dataloader:
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
            entropies.extend(entropy.cpu().numpy())

    entropies = np.array(entropies)

    # Map entropies back to unlabeled indices
    sorted_idx = np.argsort(-entropies)[:n]
    return unlabeled_idx[sorted_idx]

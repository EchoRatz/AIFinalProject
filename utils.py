import csv
import os
import yaml


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    assert cfg is not None, "config.yaml is empty or invalid"
    return cfg


def init_logger(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "labeled_samples", "accuracy"])


def log_row(path, round_id, labeled_count, accuracy):
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([round_id, labeled_count, accuracy])
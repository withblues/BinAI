from collections import Counter
from datasets import load_from_disk
import argparse
import random
import json
import os

PROJECT_LANGS = {
    "clamav": "C",
    "curl": "C",
    "nmap": "C++",
    "openssl": "C",
    "unrar": "C++",
    "z3": "C++",
    "zlib": "C",
}

def compute_cross_project_split(dataset, seed=42, output_file="cross_project_split.json"):
    random.seed(seed)
    
    projects = dataset["project"]
    
    # count rows per project
    project_counts = Counter(projects)
    print("project counts:", project_counts)
    
    # sort projects by num counts
    sorted_projects = sorted(PROJECT_LANGS.keys(), key=lambda p: project_counts[p])
    
    # assign val/test to smallest projects ensuring both C and C++ in each
    val_projects, test_projects = [], []
    
    for p in sorted_projects:
        lang = PROJECT_LANGS[p]
        if len(val_projects) < 2 and lang not in [PROJECT_LANGS[pr] for pr in val_projects]:
            val_projects.append(p)
        elif len(test_projects) < 2 and lang not in [PROJECT_LANGS[pr] for pr in test_projects]:
            test_projects.append(p)
    
    # create train split
    train_projects = [p for p in PROJECT_LANGS.keys() if p not in val_projects + test_projects]
    
    # get indices
    train_idx = [i for i, p in enumerate(projects) if p in train_projects]
    val_idx = [i for i, p in enumerate(projects) if p in val_projects]
    test_idx = [i for i, p in enumerate(projects) if p in test_projects]
    
    # Save indices
    with open(output_file, "w") as f:
        json.dump({"train": train_idx, "val": val_idx, "test": test_idx}, f)
    
    print(f"Split done. Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    print("Val projects:", val_projects)
    print("Test projects:", test_projects)
    
    return train_idx, val_idx, test_idx

def compute_cross_binary_split(dataset, seed=42, output_file="cross_binary_split.json"):
    random.seed(seed)
    
    binaries = list(set(dataset["binary_name"]))
    random.shuffle(binaries)
    
    
    n = len(binaries)
    train_bins = binaries[: int(n * 0.8)]
    val_bins = binaries[int(n * 0.8): int(n * 0.9)]
    test_bins = binaries[int(n * 0.9):]
    
    train_idx = [i for i, b in enumerate(dataset["binary_name"]) if b in train_bins]
    val_idx = [i for i, b in enumerate(dataset["binary_name"]) if b in val_bins]
    test_idx = [i for i, b in enumerate(dataset["binary_name"]) if b in test_bins]
    
    # Save indices
    with open(output_file, "w") as f:
        json.dump({"train": train_idx, "val": val_idx, "test": test_idx}, f)
    
    print(f"Cross-binary split done. Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    return train_idx, val_idx, test_idx

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Creating Train/Val/Test Split")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)

    args = parser.parse_args()
    dataset = load_from_disk(args.data_dir)

    # cross project split
    compute_cross_project_split(dataset, output_file=os.path.join(args.output_dir, "cross_project_split.json"))

    # cross binary split
    compute_cross_binary_split(dataset, output_file=os.path.join(args.output_dir, "cross_binary_split.json"))


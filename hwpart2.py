# Decision Tree (manual) for Study Hours vs Pass/Fail
# Uses Gini impurity and finds best split (no sklearn / no pandas)

from typing import List, Tuple, Any
from collections import Counter

# dataset as list of [feature, label]
dataset: List[List[Any]] = [
    [2, 0],
    [4, 0],
    [6, 1],
    [8, 1],
    [10, 1]
]

def gini(groups: List[List[List[Any]]], classes: List[Any]) -> float:
    total = sum(len(group) for group in groups)
    if total == 0:
        return 0.0
    gini_value = 0.0
    for group in groups:
        size = len(group)
        if size == 0:
            continue
        score = 0.0
        # proportion of each class in this group
        labels = [row[-1] for row in group]
        for class_val in classes:
            p = labels.count(class_val) / size
            score += p * p
        gini_value += (1.0 - score) * (size / total)
    return gini_value

def test_split(index: int, split_value: float, data: List[List[Any]]) -> Tuple[List[List[Any]], List[List[Any]]]:
    left, right = [], []
    for row in data:
        if row[index] < split_value:
            left.append(row)
        else:
            right.append(row)
    return left, right

def get_candidate_splits(data: List[List[Any]], index: int) -> List[float]:
    # Use midpoints between sorted unique values as candidate thresholds
    values = sorted({row[index] for row in data})
    candidates = []
    for i in range(len(values) - 1):
        mid = (values[i] + values[i+1]) / 2.0
        candidates.append(mid)
    # Also include splitting exactly at min and max +/- small epsilon if desired (optional)
    return candidates

def get_split(data: List[List[Any]]):
    class_values = sorted(set(row[-1] for row in data))
    best_index, best_value, best_score, best_groups = None, None, float('inf'), None

    n_features = len(data[0]) - 1  # number of features (here 1)
    for index in range(n_features):
        candidates = get_candidate_splits(data, index)
        for cand in candidates:
            groups = test_split(index, cand, data)
            g = gini(groups, class_values)
            print(f"Trying split at feature[{index}] < {cand:.2f}  --> Gini = {g:.4f}")
            if g < best_score:
                best_index, best_value, best_score, best_groups = index, cand, g, groups

    return {
        'index': best_index,
        'value': best_value,
        'score': best_score,
        'groups': best_groups
    }

if __name__ == "__main__":
    split = get_split(dataset)
    print("\nBest split found:")
    print(f" Feature index: {split['index']}")
    print(f" Split threshold: {split['value']}")
    print(f" Gini impurity at split: {split['score']:.4f}")

    left, right = split['groups']
    print("\n Left group (feature < threshold):", left)
    print(" Right group (feature >= threshold):", right)

"""
Random Forest from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - impurity
import numpy as np

def impurity(labels):
    if len(labels) == 0:
        return 0.0

    _, counts = np.unique(labels, return_counts=True)

    if len(counts) == 1:
        return 0.0

    p = counts / len(labels)
    return float(1 - np.sum(p ** 2))

# Step 2 - split_dataset
import numpy as np

def split_dataset(features, labels, feature_index, threshold):
    col = features[:, feature_index]
    mask = col <= threshold

    left_features = features[mask]
    left_labels = labels[mask]

    right_features = features[~mask]
    right_labels = labels[~mask]

    return left_features, left_labels, right_features, right_labels

# Step 3 - split_score
def split_score(parent_labels, left_labels, right_labels):
    n = len(parent_labels)

    w_left = len(left_labels) / n
    w_right = len(right_labels) / n

    parent_impurity = impurity(parent_labels)
    left_impurity = impurity(left_labels)
    right_impurity = impurity(right_labels)

    return parent_impurity - (
        w_left * left_impurity +
        w_right * right_impurity
    )

# Step 4 - best_split
import numpy as np

def best_split(features, labels, feature_indices):
    best = {
        "feature_index": None,
        "threshold": None,
        "score": 0.0
    }

    for fi in feature_indices:

        values = np.unique(features[:, fi])

        if len(values) < 2:
            continue

        thresholds = (values[:-1] + values[1:]) / 2

        for t in thresholds:

            left_X, left_y, right_X, right_y = split_dataset(
                features,
                labels,
                fi,
                t
            )

            # Skip invalid splits
            if len(left_y) == 0 or len(right_y) == 0:
                continue

            score = split_score(labels, left_y, right_y)

            if score > best["score"]:
                best["feature_index"] = fi
                best["threshold"] = t
                best["score"] = score

    return best

# Step 5 - should_stop
def should_stop(labels, depth, max_depth, min_samples_split):
    return (
        len(np.unique(labels)) == 1
        or depth >= max_depth
        or len(labels) < min_samples_split
    )

# Step 6 - leaf_prediction
import numpy as np

def leaf_prediction(labels):
    values, counts = np.unique(labels, return_counts=True)
    return int(values[np.argmax(counts)])

# Step 7 - build_tree
import numpy as np

def build_tree(
    features,
    labels,
    depth=0,
    max_depth=10,
    min_samples_split=2,
    feature_subset=None,
):
    # 1. Check stopping conditions
    if should_stop(labels, depth, max_depth, min_samples_split):
        return {
            "leaf": True,
            "prediction": leaf_prediction(labels),
        }

    # 2. Candidate features
    if feature_subset is None:
        candidate_features = list(range(features.shape[1]))
    else:
        candidate_features = list(feature_subset)

    # 3. Find the best split
    split = best_split(features, labels, candidate_features)

    # No useful split
    if split["feature_index"] is None:
        return {
            "leaf": True,
            "prediction": leaf_prediction(labels),
        }

    # 4. Split the data
    left_X, left_y, right_X, right_y = split_dataset(
        features,
        labels,
        split["feature_index"],
        split["threshold"],
    )

    # Safety check
    if len(left_y) == 0 or len(right_y) == 0:
        return {
            "leaf": True,
            "prediction": leaf_prediction(labels),
        }

    # 5. Recursively build children
    left_tree = build_tree(
        left_X,
        left_y,
        depth + 1,
        max_depth,
        min_samples_split,
        feature_subset,
    )

    right_tree = build_tree(
        right_X,
        right_y,
        depth + 1,
        max_depth,
        min_samples_split,
        feature_subset,
    )

    # 6. Return internal node
    return {
        "leaf": False,
        "feature_index": split["feature_index"],
        "threshold": split["threshold"],
        "left": left_tree,
        "right": right_tree,
    }

# Step 8 - predict_example_tree (not yet solved)
# TODO: implement

# Step 9 - predict_tree (not yet solved)
# TODO: implement

# Step 10 - bootstrap_sample (not yet solved)
# TODO: implement

# Step 11 - feature_subset (not yet solved)
# TODO: implement

# Step 12 - train_forest (not yet solved)
# TODO: implement

# Step 13 - combine_predictions (not yet solved)
# TODO: implement

# Step 14 - predict_forest (not yet solved)
# TODO: implement

# Step 15 - accuracy (not yet solved)
# TODO: implement


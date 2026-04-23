import random
import yaml
import os

def create_crossfold_splits(videoDirs_out, window_size):
    """
    Creates cross-fold splits for training and validation.

    Args:
        videoDirs_out (list): List of dataset directories.
        window_size (int): Number of directories to use as validation in each fold.

    Returns:
        list: A list of dictionaries, each containing 'train' and 'val' keys.
    """
    # Shuffle the directories
    random.shuffle(videoDirs_out)

    # Create the splits
    splits = []
    for i in range(len(videoDirs_out)):
        # Determine the validation set based on the window size
        val_dirs = videoDirs_out[i:i + window_size]
        train_dirs = [d for d in videoDirs_out if d not in val_dirs]

        # Add the split to the list
        splits.append({'train': train_dirs, 'val': val_dirs})

        # Stop if the window exceeds the list length
        if i + window_size >= len(videoDirs_out):
            break

    return splits


def save_splits_to_yaml(splits, output_file):
    """
    Saves cross-fold splits to a YAML file with the structure 'fold X: - train - val'.
    If the file already exists, creates a new file with a different name.

    Args:
        splits (list): A list of dictionaries containing 'train' and 'val' keys.
        output_file (str): Path to the YAML file where splits will be saved.
    """
    # Prepare the data with fold structure
    folds = {f"fold {i + 1}": split for i, split in enumerate(splits)}

    # Check if the file exists
    base_name, ext = os.path.splitext(output_file)
    counter = 1
    while os.path.exists(output_file):
        # Create a new file name with a counter
        output_file = f"{base_name}_{counter}{ext}"
        counter += 1

    # Save the splits to the YAML file
    with open(output_file, 'w') as file:
        yaml.dump(folds, file, default_flow_style=False)

    print(f"Splits saved to {output_file}")


# Example usage
# videoDirs_out = ['dir1', 'dir2', 'dir3', 'dir4', 'dir5']
# window_size = 1
# splits = create_crossfold_splits(videoDirs_out, window_size)
# save_splits_to_yaml(splits, 'crossfold_splits.yaml')
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import load_hsi_data, DATASET_CLASS_NAMES

def generate_split_summary(seed=42):
    """
    Generates a CSV file summarizing the number of train/test samples for each class
    in each dataset, based on the project's splitting logic.
    """
    datasets = ['IndianPines', 'PaviaUniversity', 'Houston', 'WHU-Hi-HanChuan']
    
    # SOTA-compliant training ratios as defined in demo.py
    train_ratios = {
        'IndianPines': 0.03,
        'PaviaUniversity': 0.005,
        'Houston': 0.01,
        'WHU-Hi-HanChuan': 0.005
    }

    summary_data = []

    for dataset_name in datasets:
        print(f"Processing dataset: {dataset_name}...")

        # Load ground truth labels and class names
        _, gt_labels = load_hsi_data(dataset_name)
        class_names = DATASET_CLASS_NAMES.get(dataset_name, [])
        
        # Get labels of all labeled pixels (ignore background where label is 0)
        labeled_pixels_indices = np.where(gt_labels > 0)
        all_labels = gt_labels[labeled_pixels_indices]
        
        # Get unique classes and their total counts
        unique_labels, total_counts = np.unique(all_labels, return_counts=True)
        
        # Simulate the split to get the counts without actually splitting the data
        # We create a dummy array of indices to split
        indices = np.arange(len(all_labels))
        train_ratio = train_ratios[dataset_name]
        test_ratio = 1 - train_ratio

        # Perform a stratified split on the indices
        train_indices, _ = train_test_split(
            indices,
            test_size=test_ratio,
            random_state=seed,
            stratify=all_labels
        )
        
        # Get the labels corresponding to the training set
        train_labels = all_labels[train_indices]
        
        # Count the number of training samples for each class
        _, train_counts = np.unique(train_labels, return_counts=True)

        # The unique labels from the full set and training set should align
        # but it's safer to build a dictionary for lookup
        train_counts_dict = dict(zip(np.unique(train_labels), train_counts))

        # Initialize dataset-level counters
        total_samples_dataset, train_samples_dataset, test_samples_dataset = 0, 0, 0

        for i, label_id in enumerate(unique_labels):
            total_samples = total_counts[i]
            # Use .get(label_id, 0) in case a class has 0 samples in the train set
            train_samples = train_counts_dict.get(label_id, 0)
            test_samples = total_samples - train_samples
            
            # Update dataset-level counters
            total_samples_dataset += total_samples
            train_samples_dataset += train_samples
            test_samples_dataset += test_samples

            # Get class name, fallback to "Class X" if not found
            class_name = class_names[label_id - 1] if (label_id - 1) < len(class_names) else f"Class {label_id}"

            summary_data.append({
                'Dataset': dataset_name,
                'Class Name': class_name,
                'Total Samples': total_samples,
                'Train Samples': train_samples,
                'Test Samples': test_samples
            })
        
        # Add the total row for the current dataset
        summary_data.append({
            'Dataset': dataset_name,
            'Class Name': 'Total',
            'Total Samples': total_samples_dataset,
            'Train Samples': train_samples_dataset,
            'Test Samples': test_samples_dataset
        })

    # Create DataFrame and save to CSV
    df = pd.DataFrame(summary_data)
    output_filename = 'dataset_split_summary.csv'
    df.to_csv(output_filename, index=False)
    
    print(f"\nSuccessfully generated split summary file: {output_filename}")
    print(f"\n--- Summary for {datasets[0]} ---")
    print(df[df['Dataset'] == datasets[0]].to_string())


if __name__ == '__main__':
    generate_split_summary()

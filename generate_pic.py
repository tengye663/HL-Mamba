# 文件名: generate_pic.py
# (此版本已修改，会接收 demo.py 传来的真实指标并写入 .txt 文件)

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import torch.utils.data as Data
from tqdm import tqdm
# Import pad_with_zeros from utils
from utils import pad_with_zeros, output_metric

# --- Helper class to load data for map generation ---
class _TestDS(torch.utils.data.Dataset):
    def __init__(self, padded_image, coords, labels, patch_size, pca_bands):
        self.padded_image = torch.from_numpy(padded_image).float()
        self.coords = coords
        self.labels = torch.from_numpy(labels).long()
        self.patch_size = patch_size
        self.pca_bands = pca_bands

    def __getitem__(self, index):
        r, c = self.coords[index]
        
        # Real-time patch slicing from padded image
        patch = self.padded_image[r:r + self.patch_size, c:c + self.patch_size, :]
        
        # Reshape and transpose to (1, bands, height, width)
        # The patch is already a tensor, so we don't need from_numpy
        patch = patch.reshape(self.patch_size, self.patch_size, self.pca_bands, 1)
        patch = patch.permute(3, 2, 0, 1)
        
        # Return 0-indexed label for the model
        return patch, self.labels[index] - 1

    def __len__(self):
        return len(self.coords)

# --- Helper function to run prediction (matches author's test function) ---
def _run_prediction(net, test_loader, device):
    net.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Running predictions for map"):
            inputs = inputs.to(device)
            outputs = net(inputs)
            outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
            all_preds.extend(outputs)
            all_targets.extend(labels.numpy())
    # Return only predictions, as targets are not needed for metrics here
    return np.array(all_preds), np.array(all_targets)

# --- Helper function to rebuild map (matches author's get_classification_map) ---
def _build_prediction_map(y_pred, gt_labels):
    """
    Rebuilds the full-size map from the prediction list (which only has labeled pixels).
    """
    height = gt_labels.shape[0]
    width = gt_labels.shape[1]
    cls_labels_map = np.zeros((height, width))
    k = 0 # Index for y_pred
    for i in range(height):
        for j in range(width):
            target = int(gt_labels[i, j])
            if target == 0:
                continue
            else:
                # Assign the prediction, add 1 (since preds are 0-indexed)
                cls_labels_map[i, j] = y_pred[k] + 1
                k += 1
    return cls_labels_map

# --- Helper function for colormap (matches author's list_to_colormap) ---
# NOTE: This is hardcoded for Indian Pines (16 classes) as in the author's file.
def _list_to_colormap(x_list):
    y = np.zeros((x_list.shape[0], 3))
    for index, item in enumerate(x_list):
        if item == 0:
            y[index] = np.array([0, 0, 0]) / 255.
        elif item == 1:
            y[index] = np.array([147, 67, 46]) / 255.
        elif item == 2:
            y[index] = np.array([0, 0, 255]) / 255.
        elif item == 3:
            y[index] = np.array([255, 100, 0]) / 255.
        elif item == 4:
            y[index] = np.array([0, 255, 123]) / 255.
        elif item == 5:
            y[index] = np.array([164, 75, 155]) / 255.
        elif item == 6:
            y[index] = np.array([101, 174, 255]) / 255.
        elif item == 7:
            y[index] = np.array([118, 254, 172]) / 255.
        elif item == 8:
            y[index] = np.array([60, 91, 112]) / 255.
        elif item == 9:
            y[index] = np.array([255, 255, 0]) / 255.
        elif item == 10:
            y[index] = np.array([255, 255, 125]) / 255.
        elif item == 11:
            y[index] = np.array([255, 0, 255]) / 255.
        elif item == 12:
            y[index] = np.array([100, 0, 255]) / 255.
        elif item == 13:
            y[index] = np.array([0, 172, 254]) / 255.
        elif item == 14:
            y[index] = np.array([0, 255, 0]) / 255.
        elif item == 15:
            y[index] = np.array([171, 175, 80]) / 255.
        elif item == 16:
            y[index] = np.array([101, 193, 60]) / 255.
        else: # Fallback for other datasets
             y[index] = np.array([item * 20 % 255, item * 50 % 255, item * 80 % 255]) / 255.
    return y

# --- Helper function to save map (matches author's classification_map) ---
def _save_classification_map(map_image, ground_truth, dpi, save_path):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(ground_truth.shape[1] * 2.0 / dpi, ground_truth.shape[0] * 2.0 / dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)
    ax.imshow(map_image)
    fig.savefig(save_path, dpi=dpi)
    plt.close()

# --- MODIFICATION: Add 'final_metrics' as a new argument ---
def generate_classification_map(model, pca_data, gt_labels, patch_size, batch_size, device, dataset_name, final_metrics):
    """
    Generates maps and saves the *true test metrics* passed from the training script.
    """
    pca_bands = pca_data.shape[2]
    margin = (patch_size - 1) // 2

    # 1. Pad the HSI data
    padded_pca_data = pad_with_zeros(pca_data, margin=margin)

    # 2. Find coordinates of all labeled pixels
    labeled_coords = np.transpose(np.nonzero(gt_labels))
    all_labels = gt_labels[labeled_coords[:, 0], labeled_coords[:, 1]]

    # 3. Create a DataLoader for all labeled pixels using the new on-the-fly patch generation
    all_dataset = _TestDS(padded_pca_data, labeled_coords, all_labels, patch_size, pca_bands)
    all_loader = torch.utils.data.DataLoader(
        dataset=all_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    # 4. Run prediction (ONLY for the visual map)
    # We don't need y_true from here, as metrics are pre-calculated
    y_pred, _ = _run_prediction(model, all_loader, device) # <-- Changed: only need y_pred

    # 5. Rebuild the full classification map
    prediction_map = _build_prediction_map(y_pred, gt_labels)

    # 6. Create colormap images
    flat_pred_map = np.ravel(prediction_map)
    flat_gt_map = gt_labels.flatten()

    pred_list_color = _list_to_colormap(flat_pred_map)
    gt_list_color = _list_to_colormap(flat_gt_map)

    pred_img = np.reshape(pred_list_color, (gt_labels.shape[0], gt_labels.shape[1], 3))
    gt_img = np.reshape(gt_list_color, (gt_labels.shape[0], gt_labels.shape[1], 3))

    # 7. Save the maps separately
    save_dir = 'classification_maps'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    pred_save_path = os.path.join(save_dir, f'{dataset_name}_predictions.png')
    gt_save_path = os.path.join(save_dir, f'{dataset_name}_gt.png')

    _save_classification_map(pred_img, gt_labels, 300, pred_save_path)
    _save_classification_map(gt_img, gt_labels, 300, gt_save_path)

    print(f"Classification maps saved to: {save_dir}")

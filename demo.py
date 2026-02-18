import torch
import argparse
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
import time
import os
import scipy.io as sio
from thop import profile
from Ours import CDCNN_network
from utils import (
    setup_seed, load_hsi_data, apply_pca, pad_with_zeros,
    split_data, AverageMeter, output_metric, DATASET_CLASS_NAMES
)
from generate_pic import generate_classification_map


# --- Argument Parsing ---
parser = argparse.ArgumentParser("SSFTT for HSI Classification")
parser.add_argument('--gpu_id', default='0', help='GPU ID')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--dataset', choices=['IndianPines', 'PaviaUniversity', 'Houston', 'WHU-Hi-HanChuan'], default='WHU-Hi-HanChuan', help='Dataset selection')
parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('--patch_size', type=int, default=11, help='Image patch size')
parser.add_argument('--pca_bands', type=int, default=30, help='Number of bands after PCA')

# SSFTT Model Parameters
parser.add_argument('--dim', type=int, default=64, help='Transformer dimension')
parser.add_argument('--depth', type=int, default=1, help='Transformer depth')
parser.add_argument('--heads', type=int, default=4, help='Number of attention heads')
parser.add_argument('--mlp_dim', type=int, default=8, help='MLP hidden dimension')
parser.add_argument('--num_tokens', type=int, default=4, help='Number of tokens')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate in Transformer')
parser.add_argument('--emb_dropout', type=float, default=0.1, help='Embedding dropout rate')
parser.add_argument('--conv3d_out_channels', type=int, default=8, help='Output channels of the 3D convolution layer')
parser.add_argument('--runs', type=int, default=10, help='Number of runs for averaging results')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

# --- Custom Dataset Class ---
class HsiDataset(Data.Dataset):
    def __init__(self, padded_image, coords, labels, patch_size, pca_bands):
        self.padded_image = torch.from_numpy(padded_image).float()
        self.coords = coords
        self.labels = torch.from_numpy(labels - 1).long()
        self.patch_size = patch_size
        self.pca_bands = pca_bands
        self.margin = (patch_size - 1) // 2

    def __getitem__(self, index):
        r, c = self.coords[index]
        
        # Real-time patch slicing
        patch = self.padded_image[r:r + self.patch_size, c:c + self.patch_size, :]
        
        # Reshape and transpose to (1, bands, height, width)
        # The patch is already a tensor, so we don't need from_numpy
        patch = patch.reshape(self.patch_size, self.patch_size, self.pca_bands, 1)
        patch = patch.permute(3, 2, 0, 1)
        
        return patch, self.labels[index]

    def __len__(self):
        return len(self.coords)

def main():
    # Lists to store metrics for averaging
    all_oa, all_aa, all_kappa = [], [], []
    all_test_times = []
    all_aa_per_class = []
    all_matrices = []

    # --- Create a directory for run-specific results ---
    results_dir = f"{args.dataset}_results"
    os.makedirs(results_dir, exist_ok=True)

    for i in range(args.runs):
        print(f"--- Run {i+1}/{args.runs} ---")
        
        # Set a different seed for each run
        run_seed = args.seed + i
        setup_seed(run_seed)

        # --- 1. Data Loading and Preprocessing ---
        print(f"Loading {args.dataset} dataset...")
        hsi_data, gt_labels = load_hsi_data(args.dataset)
        num_classes = np.max(gt_labels)
        
        print(f"Applying PCA, reducing to {args.pca_bands} bands...")
        hsi_pca = apply_pca(hsi_data, args.pca_bands)
        

        # 定义 SOTA-Compliant 的训练/测试比例
        if args.dataset == 'IndianPines':
            train_ratio, test_ratio = 0.03, 0.97
        elif args.dataset == 'PaviaUniversity':
            train_ratio, test_ratio = 0.005, 0.995
        elif args.dataset == 'Houston':
            train_ratio, test_ratio = 0.01, 0.99
        elif args.dataset == 'WHU-Hi-HanChuan':
            train_ratio, test_ratio = 0.005, 0.995
        else: # Fallback
            train_ratio, test_ratio = 0.10, 0.90

        print("Padding the image...")
        margin = (args.patch_size - 1) // 2
        padded_hsi = pad_with_zeros(hsi_pca, margin=margin)

        print("Extracting coordinates of labeled pixels...")
        # Find coordinates of all labeled pixels
        labeled_pixels = np.transpose(np.nonzero(gt_labels))
        all_labels = gt_labels[labeled_pixels[:, 0], labeled_pixels[:, 1]]

        print(f"Splitting coordinates: {train_ratio*100}% train, {test_ratio*100}% test...")
        train_coords, test_coords, y_train, y_test = split_data(
            labeled_pixels, all_labels, train_ratio, test_ratio, run_seed
        )

        # Create datasets
        train_dataset = HsiDataset(padded_hsi, train_coords, y_train, args.patch_size, args.pca_bands)
        test_dataset = HsiDataset(padded_hsi, test_coords, y_test, args.patch_size, args.pca_bands)
        
        train_loader = Data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        test_loader = Data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        
        print(f"Data preparation finished. Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

        # --- 2. Model, Loss, and Optimizer Initialization ---
        model = CDCNN_network(
            args.pca_bands,
            num_classes
            # patch_size=args.patch_size,
            # dim=args.dim,
            # depth=args.depth,
            # heads=args.heads,
            # mlp_dim=args.mlp_dim,
            # num_tokens=args.num_tokens, 
            # dropout=args.dropout,
            # emb_dropout=args.emb_dropout,
            # conv3d_out_channels=args.conv3d_out_channels
        ).cuda()
        
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        # --- 3. Calculate FLOPs and Params ---
        if i == 0: # Calculate FLOPs and Params only on the first run
            # Create a dummy input tensor with the correct shape
            # Shape: (batch_size, 1, pca_bands, patch_size, patch_size)
            dummy_input = torch.randn(1, 1, args.pca_bands, args.patch_size, args.patch_size).cuda()
            flops, params = profile(model, inputs=(dummy_input,))
        
        print(f"Model initialized.")
        print(f"  - Parameters: {params / 1e6:.4f}M")
        print(f"  - FLOPs: {flops / 1e9:.4f}G")

        print("Starting training...")
        
        # 您的代码是保存最后一轮的模型，我们先按这个逻辑来
        for epoch in range(args.epochs):
            model.train()
            train_loss_avg = AverageMeter()
            for batch_data, batch_target in train_loader:
                batch_data, batch_target = batch_data.cuda(), batch_target.cuda()
                
                optimizer.zero_grad()
                output = model(batch_data)
                loss = criterion(output, batch_target)
                loss.backward()
                optimizer.step()
                
                train_loss_avg.update(loss.item(), batch_data.size(0))
                
            print(f"Epoch [{epoch+1}/{args.epochs}] | Train Loss: {train_loss_avg.avg:.4f}")

        print(f"Training finished.")
        
        if not os.path.exists('Wights'):
            os.makedirs('Wights')
        final_model_path = f'Wights/final_model_{args.dataset}.pth'
        torch.save(model.state_dict(), final_model_path)
        print(f"Final model (epoch {args.epochs}) saved to {final_model_path}")

        print("Evaluating final model on test set...")
        
        tic = time.time()
        model.eval() 
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch_data, batch_target in test_loader:
                batch_data, batch_target = batch_data.cuda(), batch_target.cuda()
                output = model(batch_data)
                _, preds = torch.max(output, 1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(batch_target.cpu().numpy())

        toc = time.time()
        
        test_time = toc - tic
        oa, kappa, aa, aa_per_class, matrix = output_metric(np.array(all_targets), np.array(all_preds))
        
        print(f"--- Run {i+1} Test Metrics ---")
        print(f"Test Time: {test_time:.2f}s")
        print(f"OA: {oa:.4f}, Kappa: {kappa:.4f}, AA: {aa:.4f}")

        # --- Save metrics for the current run ---
        run_stats_filename = os.path.join(results_dir, f"run_{i+1}_stats.txt")
        with open(run_stats_filename, 'w') as f:
            f.write(f"--- Performance Stats for {args.dataset} (Run {i+1}) ---\n")
            f.write(f"Seed: {run_seed}\n")
            f.write(f"Parameters (M): {params / 1e6:.4f}\n")
            f.write(f"FLOPs (G): {flops / 1e9:.4f}\n")
            f.write(f"Testing Time (seconds): {test_time:.4f}\n\n")
            
            f.write(f"--- Classification Metrics (Test Set) ---\n")
            f.write(f"Overall Accuracy (OA): {oa:.4f}\n")
            f.write(f"Average Accuracy (AA): {aa:.4f}\n")
            f.write(f"Cohen's Kappa (Kappa): {kappa:.4f}\n\n")
            
            class_names = DATASET_CLASS_NAMES.get(args.dataset, [])
            f.write("--- Per-Class Accuracy ---\n")
            for j, class_acc in enumerate(aa_per_class):
                class_name = class_names[j] if j < len(class_names) else f"Class {j+1}"
                f.write(f"{class_name}: {class_acc:.4f}\n")
            
            f.write("\n--- Confusion Matrix ---\n")
            f.write(np.array2string(matrix))
        print(f"Stats for run {i+1} saved to {run_stats_filename}")

        # Store metrics for averaging
        all_oa.append(oa)
        all_aa.append(aa)
        all_kappa.append(kappa)
        all_test_times.append(test_time)
        all_aa_per_class.append(aa_per_class)
        all_matrices.append(matrix)

        # Only generate map for the last run
        if i == args.runs - 1:
            print("Generating classification map for the last run...")
            final_metrics = (oa, kappa, aa, aa_per_class)
            generate_classification_map(
                model,
                hsi_pca,
                gt_labels,
                args.patch_size,
                args.batch_size,
                'cuda',
                args.dataset,
                final_metrics
            )

    # --- Final Reporting ---
    class_names = DATASET_CLASS_NAMES.get(args.dataset, [])
    if args.runs > 1:
        print("\n--- Averaged Results ---")
        print(f"OA: {np.mean(all_oa):.4f} ± {np.std(all_oa):.4f}")
        print(f"AA: {np.mean(all_aa):.4f} ± {np.std(all_aa):.4f}")
        print(f"Kappa: {np.mean(all_kappa):.4f} ± {np.std(all_kappa):.4f}")
        print(f"Average Test Time: {np.mean(all_test_times):.4f}s")

        # Calculate per-class average and std
        all_aa_per_class = np.array(all_aa_per_class)
        mean_per_class = np.mean(all_aa_per_class, axis=0)
        std_per_class = np.std(all_aa_per_class, axis=0)

        # Calculate average confusion matrix
        mean_matrix = np.mean(all_matrices, axis=0)

        # Save averaged stats
        stats_filename = f"{args.dataset}_avg_stats.txt"
        with open(stats_filename, 'w') as f:
            f.write(f"--- Averaged Performance Stats for {args.dataset} ({args.runs} runs) ---\n")
            f.write(f"Parameters (M): {params / 1e6:.4f}\n")
            f.write(f"FLOPs (G): {flops / 1e9:.4f}\n")
            f.write(f"Average Test Time (seconds): {np.mean(all_test_times):.4f}\n\n")
            
            f.write(f"--- Averaged Classification Metrics (Test Set) ---\n")
            f.write(f"Overall Accuracy (OA): {np.mean(all_oa):.4f} ± {np.std(all_oa):.4f}\n")
            f.write(f"Average Accuracy (AA): {np.mean(all_aa):.4f} ± {np.std(all_aa):.4f}\n")
            f.write(f"Cohen's Kappa (Kappa): {np.mean(all_kappa):.4f} ± {np.std(all_kappa):.4f}\n\n")

            f.write("--- Per-Class Accuracy ---\n")
            for i, (mean_acc, std_acc) in enumerate(zip(mean_per_class, std_per_class)):
                class_name = class_names[i] if i < len(class_names) else f"Class {i+1}"
                f.write(f"{class_name}: {mean_acc:.4f} ± {std_acc:.4f}\n")
            
            f.write("\n--- Averaged Confusion Matrix ---\n")
            f.write(np.array2string(mean_matrix, formatter={'float_kind':lambda x: "%.2f" % x}))

        print(f"Averaged stats saved to {stats_filename}")
    else:
        # Save stats for a single run
        stats_filename = f"{args.dataset}_stats.txt"
        with open(stats_filename, 'w') as f:
            f.write(f"--- Performance Stats for {args.dataset} ---\n")
            f.write(f"Parameters (M): {params / 1e6:.4f}\n")
            f.write(f"FLOPs (G): {flops / 1e9:.4f}\n")
            f.write(f"Testing Time (seconds): {all_test_times[0]:.4f}\n\n")
            
            f.write(f"--- Classification Metrics (Test Set) ---\n")
            f.write(f"Overall Accuracy (OA): {all_oa[0]:.4f}\n")
            f.write(f"Average Accuracy (AA): {all_aa[0]:.4f}\n")
            f.write(f"Cohen's Kappa (Kappa): {all_kappa[0]:.4f}\n\n")
            
            f.write("--- Per-Class Accuracy ---\n")
            for i, class_acc in enumerate(all_aa_per_class[0]):
                class_name = class_names[i] if i < len(class_names) else f"Class {i+1}"
                f.write(f"{class_name}: {class_acc:.4f}\n")
            
            f.write("\n--- Confusion Matrix ---\n")
            f.write(np.array2string(matrix)) # Note: matrix is from the last run
            
        print(f"Stats and metrics saved to {stats_filename}")


if __name__ == '__main__':
    main()

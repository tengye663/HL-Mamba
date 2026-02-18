import torch
import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score
import torch.backends.cudnn as cudnn

def setup_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    cudnn.benchmark = False

def apply_pca(data, num_components):
    """Apply PCA to HSI data, matching author's 'whiten=True'"""
    h, w, d = data.shape
    data_reshaped = data.reshape(-1, d)
    # Author's code uses whiten=True
    pca = PCA(n_components=num_components, whiten=True) 
    data_pca = pca.fit_transform(data_reshaped)
    return data_pca.reshape(h, w, num_components)

def pad_with_zeros(X, margin=2):
    """Pad HSI data with zeros (matches author's padWithZeros)"""
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX

def create_patches(X, y, patch_size=5, mask=None):
    """
    Create image patches.
    If a mask is provided, only create patches for pixels where mask > 0.
    """
    margin = int((patch_size - 1) / 2)
    zero_padded_X = pad_with_zeros(X, margin=margin)
    
    # If no mask is provided, create one from the ground truth `y`
    if mask is None:
        mask = y

    # Pre-allocate memory for patches based on the number of labeled pixels in the mask
    num_patches = np.count_nonzero(mask)
    if num_patches == 0:
        return np.array([]), np.array([])
        
    patches_data = np.zeros((num_patches, patch_size, patch_size, X.shape[2]))
    patches_labels = np.zeros(num_patches)
    
    patch_index = 0
    for r in range(margin, zero_padded_X.shape[0] - margin):
        for c in range(margin, zero_padded_X.shape[1] - margin):
            # Check if the pixel is in the mask
            if mask[r - margin, c - margin] > 0:
                patch = zero_padded_X[r - margin:r + margin + 1, c - margin:c + margin + 1]
                patches_data[patch_index, :, :, :] = patch
                patches_labels[patch_index] = y[r - margin, c - margin]
                patch_index += 1
                
    return patches_data, patches_labels

def split_data(patches, labels, train_ratio, test_ratio, random_seed):
    X_train, X_test, y_train, y_test = train_test_split(
        patches, 
        labels, 
        test_size=test_ratio, 
        random_state=random_seed,
        stratify=labels
    )
    return X_train, X_test, y_train, y_test

def load_hsi_data(dataset_name, dataset_path='dataset/'):
    """Load specified HSI dataset"""
    if dataset_name == 'IndianPines':
        # Author's code uses 'Indian_pines_corrected.mat'
        data = sio.loadmat(dataset_path + 'Indian_pines_corrected.mat')['indian_pines_corrected']
        labels = sio.loadmat(dataset_path + 'Indian_pines_gt.mat')['indian_pines_gt']
    elif dataset_name == 'PaviaUniversity':
        data = sio.loadmat(dataset_path + 'PaviaU.mat')['paviaU']
        labels = sio.loadmat(dataset_path + 'PaviaU_gt.mat')['paviaU_gt']
    elif dataset_name == 'Houston':
        data = sio.loadmat(dataset_path + 'Houston.mat')['data']
        labels = sio.loadmat(dataset_path + 'Houston_gt.mat')['groundT']
    elif dataset_name == 'WHU-Hi-HanChuan':
        data = sio.loadmat(dataset_path + 'WHU_Hi_HanChuan.mat')['WHU_Hi_HanChuan']
        labels = sio.loadmat(dataset_path + 'WHU_Hi_HanChuan_gt.mat')['WHU_Hi_HanChuan_gt']
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
        
    # Convert data to float32 for processing
    return data.astype(np.float32), labels.astype(np.int32)

class AverageMeter(object):
    """Compute and store the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

def output_metric(tar, pre):
    """Calculate and return classification metrics"""
    matrix = confusion_matrix(tar, pre)
    oa = accuracy_score(tar, pre)
    kappa = cohen_kappa_score(tar, pre)
    
    # Calculate AA (Average Accuracy)
    aa_per_class = np.diag(matrix) / matrix.sum(axis=1)
    aa = np.mean(aa_per_class)
    
    return oa, kappa, aa, aa_per_class, matrix

DATASET_CLASS_NAMES = {
    "IndianPines": [
        "Alfalfa", "Corn-notill", "Corn-mintill", "Corn", "Grass-pasture",
        "Grass-trees", "Grass-pasture-mowed", "Hay-windrowed", "Oats",
        "Soybean-notill", "Soybean-mintill", "Soybean-clean", "Wheat",
        "Woods", "Buildings-Grass-Trees-Drives", "Stone-Steel-Towers"
    ],
    "PaviaUniversity": [
        "Asphalt", "Meadows", "Gravel", "Trees", "Painted metal sheets",
        "Bare Soil", "Bitumen", "Self-Blocking Bricks", "Shadows"
    ],
    "Houston": [
        "Healthy grass", "Stressed grass", "Synthetic grass", "Trees", "Soil",
        "Water", "Residential", "Commercial", "Road", "Highway", "Railway",
        "Parking Lot 1", "Parking Lot 2", "Tennis Court", "Running Track"
    ],
    "WHU-Hi-HanChuan": [
        "Strawberry", "Cowpea", "Soybean", "Sorghum", "Water-spinach",
        "Watermelon", "Greens", "Trees", "Grass", "Red-roof", "Gray-roof",
        "Plastic", "Bare-soil", "Road", "Bright-object", "Water"
    ]
}

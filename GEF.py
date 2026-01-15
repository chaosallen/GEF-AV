import cv2
import numpy as np
import os
import glob
from skimage.morphology import skeletonize
from scipy.ndimage import gaussian_filter

# ==================================================
# User configuration
# ==================================================

# Input binary vessel masks
artery_dir = './Data/OCTA500/train/Binary_A'
vein_dir   = './Data/OCTA500/train/Binary_V'

# Output root directory
save_root  = './Data/OCTA500/train'

# -------- DEF configuration --------
def_type = 'gaussian'
# Options: 'gaussian', 'linear', 'exponential', 'inverse'

def_sigma = 15
def_alpha = 0.1

# -------- OEF configuration --------
oef_type = 'cos'
# Options: 'cos', 'angle'

oef_sigma = 3

# ==================================================
# Output directories
# ==================================================
save_def_a = os.path.join(save_root, 'DEF_A')
save_def_v = os.path.join(save_root, 'DEF_V')
save_def_m = os.path.join(save_root, 'DEF_M')

save_oef_a = os.path.join(save_root, 'OEF_A')
save_oef_v = os.path.join(save_root, 'OEF_V')
save_oef_m = os.path.join(save_root, 'OEF_M')

for d in [save_def_a, save_def_v, save_def_m,
          save_oef_a, save_oef_v, save_oef_m]:
    os.makedirs(d, exist_ok=True)

# ==================================================
# DEF: Distance Energy Field
# ==================================================
def energy_gaussian(dist, sigma=20):
    return np.exp(-(dist ** 2) / (2 * sigma ** 2))


def energy_linear(dist, eps=1e-6):
    max_dist = dist.max() + eps
    return np.clip(1.0 - dist / max_dist, 0.0, 1.0)


def energy_exponential(dist, alpha=0.1):
    return np.exp(-alpha * dist)


def energy_inverse(dist, alpha=0.1):
    return 1.0 / (1.0 + alpha * dist)


def binary_to_def(binary_img,
                  mode='gaussian',
                  sigma=15,
                  alpha=0.1):
    """
    Convert binary vessel mask to Distance Energy Field (DEF).
    """
    inverted = 1 - binary_img.astype(np.uint8)
    dist = cv2.distanceTransform(inverted, cv2.DIST_L2, 3)

    if mode == 'gaussian':
        return energy_gaussian(dist, sigma)
    elif mode == 'linear':
        return energy_linear(dist)
    elif mode == 'exponential':
        return energy_exponential(dist, alpha)
    elif mode == 'inverse':
        return energy_inverse(dist, alpha)
    else:
        raise ValueError(f'Unsupported DEF type: {mode}')


def save_energy_map(energy, save_path):
    """
    Normalize energy map to [0, 255] and save as uint8 image.
    """
    energy_norm = (energy - energy.min()) / (energy.max() - energy.min() + 1e-6)
    cv2.imwrite(save_path, (energy_norm * 255).astype(np.uint8))

# ==================================================
# OEF: Orientation Energy Field
# ==================================================
def get_skeleton(binary_img):
    """
    Extract vessel skeleton from binary mask.
    """
    return skeletonize(binary_img > 0).astype(np.uint8)


def compute_skeleton_gradient(skeleton):
    """
    Compute normalized gradient of vessel skeleton.
    """
    skel_f = skeleton.astype(np.float32)
    gx = cv2.Sobel(skel_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(skel_f, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2) + 1e-8
    return gx / mag, gy / mag


def gradient_angle_cos_map(gx, gy):
    """
    Orientation encoding using cosine representation.
    """
    mag = np.sqrt(gx ** 2 + gy ** 2) + 1e-8
    cos_theta = gx / mag
    return (((cos_theta + 1.0) / 2.0) * 255).astype(np.uint8)


def gradient_angle_theta_map(gx, gy):
    """
    Orientation encoding using angle representation.
    """
    theta = np.arctan2(gy, gx)
    return (((theta + np.pi) / (2 * np.pi)) * 255).astype(np.uint8)


# ==================================================
# Main processing pipeline (GEF)
# ==================================================
file_list = glob.glob(os.path.join(artery_dir, '*.bmp'))

for filepath in file_list:
    filename = os.path.basename(filepath)

    artery = cv2.imread(os.path.join(artery_dir, filename), cv2.IMREAD_GRAYSCALE)
    vein   = cv2.imread(os.path.join(vein_dir, filename),   cv2.IMREAD_GRAYSCALE)

    if artery is None or vein is None:
        print(f"Skip {filename}, failed to read.")
        continue

    artery = (artery > 127).astype(np.uint8)
    vein   = (vein > 127).astype(np.uint8)

    # ---------------- DEF ----------------
    def_a = binary_to_def(artery, def_type, def_sigma, def_alpha)
    def_v = binary_to_def(vein,   def_type, def_sigma, def_alpha)
    def_m = def_a - def_v

    save_energy_map(def_a, os.path.join(save_def_a, filename))
    save_energy_map(def_v, os.path.join(save_def_v, filename))
    save_energy_map(def_m, os.path.join(save_def_m, filename))

    # ---------------- OEF ----------------
    skel_a = get_skeleton(artery)
    skel_v = get_skeleton(vein)

    gx_a, gy_a = compute_skeleton_gradient(skel_a)
    gx_v, gy_v = compute_skeleton_gradient(skel_v)

    gx_a = gaussian_filter(gx_a, sigma=oef_sigma)
    gy_a = gaussian_filter(gy_a, sigma=oef_sigma)
    gx_v = gaussian_filter(gx_v, sigma=oef_sigma)
    gy_v = gaussian_filter(gy_v, sigma=oef_sigma)

    gx_m = gx_a - gx_v
    gy_m = gy_a - gy_v

    if oef_type == 'cos':
        oef_a = gradient_angle_cos_map(gx_a, gy_a)
        oef_v = gradient_angle_cos_map(gx_v, gy_v)
        oef_m = gradient_angle_cos_map(gx_m, gy_m)
    else:
        oef_a = gradient_angle_theta_map(gx_a, gy_a)
        oef_v = gradient_angle_theta_map(gx_v, gy_v)
        oef_m = gradient_angle_theta_map(gx_m, gy_m)

    cv2.imwrite(os.path.join(save_oef_a, filename), oef_a)
    cv2.imwrite(os.path.join(save_oef_v, filename), oef_v)
    cv2.imwrite(os.path.join(save_oef_m, filename), oef_m)

    print(f"Processed: {filename}")

print("All GEF (DEF + OEF) maps generated successfully.")

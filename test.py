import torch
import torch.nn as nn
import logging
import sys
import os
from models.MAUNet import MAUNet
import numpy as np
import utils
import pandas as pd
from options.test_options import TestOptions
import cv2
import natsort
import BatchDataReader
import scipy.misc as misc
from tqdm import tqdm
from prefetch_generator import BackgroundGenerator


def test_net(net, device, testroot, saveroot, data_size, input_filename, label_filename, energy_filename,
             threshold=0.5):

    # === Create directories to save results ===
    save_root = os.path.join(saveroot, 'test_results')
    artery_save = os.path.join(save_root, 'artery')
    vein_save = os.path.join(save_root, 'vein')
    visual_save = os.path.join(saveroot, 'test_visuals')
    energy_save = os.path.join(save_root, 'energy')

    os.makedirs(artery_save, exist_ok=True)
    os.makedirs(vein_save, exist_ok=True)
    os.makedirs(visual_save, exist_ok=True)
    os.makedirs(energy_save, exist_ok=True)

    # === Load test dataset ===
    test_dataset = BatchDataReader.CubeDataset(testroot, data_size,
                                               input_filename, label_filename, energy_filename)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

    net.eval()  # Set model to evaluation mode
    pbar = tqdm(enumerate(BackgroundGenerator(test_loader)), total=len(test_loader))

    # Initialize metric accumulators
    metrics_sum = {k: 0.0 for k in ['Dice', 'Accuracy', 'Sensitivity', 'Specificity', 'clDice', 'HD95']}
    metrics_count = {k: 0 for k in ['Dice', 'Accuracy', 'Sensitivity', 'Specificity', 'clDice', 'HD95']}

    sample_count = 0
    valid_sample_count = 0  # Count of samples with valid metrics

    for itr, (test_images, test_labels, energy_labels, cubename) in pbar:
        test_images = test_images.to(device=device, dtype=torch.float32)
        test_labels = test_labels.permute(0, 3, 1, 2).to(device=device, dtype=torch.float32)

        with torch.no_grad():
            # Forward pass: network outputs energy map and segmentation logits
            pred_energy, pred3 = net(test_images)

        pred = torch.sigmoid(pred3)  # Convert logits to probabilities
        pred_np = pred.squeeze(0).cpu().numpy()  # [2, H, W], probability values
        label_np = test_labels.squeeze(0).cpu().numpy()  # [2, H, W], ground truth

        # Energy map prediction
        energy_np = pred_energy.squeeze(0).cpu().numpy()  # [en_channel, H, W]

        # Convert predictions and ground truth to binary masks
        pred_bin_np = (pred_np > threshold).astype(np.uint8)
        gt_bin_np = (label_np > 0.5).astype(np.uint8)

        # Compute metrics for arteries and veins
        sample_metrics = []
        for c in range(2):  # 0: artery, 1: vein
            if np.any(gt_bin_np[c]) or np.any(pred_bin_np[c]):  # Only compute if there is mask
                metrics = utils.compute_metrics(pred_bin_np[c], gt_bin_np[c])
                sample_metrics.append(metrics)

        # Accumulate valid metrics
        if sample_metrics:
            valid_sample_count += 1
            for metric in metrics_sum.keys():
                values = [m[metric] for m in sample_metrics if not np.isnan(m[metric])]
                if values:
                    avg_value = np.mean(values)
                    metrics_sum[metric] += avg_value
                    metrics_count[metric] += 1

        # === Save predicted images ===
        try:
            base_name = os.path.splitext(cubename[0])[0]

            # Get original image size from the first label (fallback to data_size)
            if len(label_filename) > 0:
                label_path = os.path.join(testroot, label_filename[0], cubename[0])
                if os.path.exists(label_path):
                    label_img = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
                    if label_img is not None:
                        original_size = (label_img.shape[1], label_img.shape[0])
                    else:
                        original_size = (data_size[1], data_size[0])
                else:
                    original_size = (data_size[1], data_size[0])
            else:
                original_size = (data_size[1], data_size[0])

            # 1. Binary masks for artery and vein
            artery_mask = (pred_np[0] >= threshold).astype(np.uint8) * 255
            vein_mask = (pred_np[1] >= threshold).astype(np.uint8) * 255

            # 2. Resize to original size
            artery_mask_resized = cv2.resize(artery_mask, original_size, interpolation=cv2.INTER_NEAREST)
            vein_mask_resized = cv2.resize(vein_mask, original_size, interpolation=cv2.INTER_NEAREST)

            # 3. Save artery and vein masks
            cv2.imwrite(os.path.join(artery_save, f"{base_name}.png"), artery_mask_resized)
            cv2.imwrite(os.path.join(vein_save, f"{base_name}.png"), vein_mask_resized)

            # 4. Resize probability maps for RGB visualization
            pred_artery_resized = cv2.resize(pred_np[0], original_size, interpolation=cv2.INTER_LINEAR)
            pred_vein_resized = cv2.resize(pred_np[1], original_size, interpolation=cv2.INTER_LINEAR)

            # 5. Create RGB overlay (R: artery, G: vein)
            BGR = np.zeros((original_size[1], original_size[0], 3), dtype=np.uint8)
            BGR[:, :, 2] = (pred_artery_resized * 255).astype(np.uint8)  # Red channel: artery
            BGR[:, :, 1] = (pred_vein_resized * 255).astype(np.uint8)    # Green channel: vein

            # 6. Save visualization
            cv2.imwrite(os.path.join(visual_save, cubename[0]), BGR)

            # 7. Save energy maps as images
            if energy_np is not None:
                energy_channels = energy_np.shape[0]
                for ch in range(energy_channels):
                    energy_channel = energy_np[ch]
                    energy_resized = cv2.resize(energy_channel, original_size, interpolation=cv2.INTER_LINEAR)
                    energy_prob = (energy_resized * 255).astype(np.uint8)
                    energy_path = os.path.join(energy_save, f"{base_name}_ch{ch}.png")
                    cv2.imwrite(energy_path, energy_prob)

        except Exception as e:
            logging.warning(f"Failed to save images for {cubename[0]}: {e}")

        sample_count += 1
        pbar.set_description(f"Testing... Sample: {sample_count}")

    # === Compute average metrics across samples ===
    avg_metrics = {}
    for metric in metrics_sum.keys():
        if metrics_count[metric] > 0:
            avg_metrics[metric] = metrics_sum[metric] / metrics_count[metric]
        else:
            avg_metrics[metric] = np.nan

    # Print final results
    print("\n" + "=" * 50)
    print(f"Testing completed! Total samples: {sample_count}, Valid samples: {valid_sample_count}")
    for metric, value in avg_metrics.items():
        if metric == 'HD95':
            print(f"{metric}: {value:.2f}" if not np.isnan(value) else f"{metric}: N/A")
        else:
            print(f"{metric}: {value:.4f}" if not np.isnan(value) else f"{metric}: N/A")
    print("=" * 50)

    return avg_metrics


if __name__ == '__main__':
    # Set logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Load test options
    opt = TestOptions().parse()

    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')

    # Initialize network
    en_channel = len(opt.energy_filename)
    net = MAUNet(in_channels=opt.in_channels, n_classes=opt.n_classes, channels=opt.channels, channels1st=en_channel)

    # Load trained best model
    bestmodelpath = os.path.join(opt.saveroot, 'best_model',
                                 natsort.natsorted(os.listdir(os.path.join(opt.saveroot, 'best_model')))[-1])
    restore_path = os.path.join(bestmodelpath, os.listdir(bestmodelpath)[0])
    net.load_state_dict(torch.load(restore_path, map_location=device))
    net.to(device=device)

    # Define datasets to test
    datasets = [
        {'name': 'OCTA500_6M', 'testroot': 'G:\\AV Segmentation\\Data\\OCTA500\\test_6M',
         'saveroot': os.path.join(opt.saveroot, 'OCTA500_6M_TEST')},
        {'name': 'OCTA500_3M', 'testroot': 'G:\\AV Segmentation\\Data\\OCTA500\\test_3M',
         'saveroot': os.path.join(opt.saveroot, 'OCTA500_3M_TEST')},
        {'name': 'OCTA500', 'testroot': 'G:\\AV Segmentation\\Data\\OCTA500\\test',
         'saveroot': os.path.join(opt.saveroot, 'OCTA500_TEST')},
    ]

    try:
        # Store results for all datasets
        all_results = {}

        for dataset in datasets:
            logging.info(f'\nTesting dataset: {dataset["name"]}')
            os.makedirs(dataset['saveroot'], exist_ok=True)

            results = test_net(
                net=net,
                device=device,
                testroot=dataset['testroot'],
                saveroot=dataset['saveroot'],
                data_size=opt.data_size,
                input_filename=opt.input_filename,
                label_filename=opt.label_filename,
                energy_filename=opt.energy_filename,
                threshold=0.5
            )

            all_results[dataset['name']] = results

        # Save summary of all datasets to CSV
        try:
            summary_data = []
            for dataset_name, metrics in all_results.items():
                row = {'dataset': dataset_name}
                row.update(metrics)
                summary_data.append(row)

            df_summary = pd.DataFrame(summary_data)
            summary_path = os.path.join(opt.saveroot, 'all_datasets_summary.csv')
            df_summary.to_csv(summary_path, index=False, float_format='%.4f')
            logging.info(f"Summary of all datasets saved to: {summary_path}")

        except Exception as e:
            logging.error(f"Error saving summary: {e}")

        print("\nAll testing completed!")

    except KeyboardInterrupt:
        logging.info('Testing interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

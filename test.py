import torch
import torch.nn as nn
import logging
import sys
import os
from models.MAUNet2 import MAUNet2
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

    # Create save directories
    save_root = os.path.join(saveroot, 'test_results')
    artery_save = os.path.join(save_root, 'artery')
    vein_save = os.path.join(save_root, 'vein')
    visual_save = os.path.join(saveroot, 'test_visuals')
    energy_save = os.path.join(save_root, 'energy')

    os.makedirs(artery_save, exist_ok=True)
    os.makedirs(vein_save, exist_ok=True)
    os.makedirs(visual_save, exist_ok=True)
    os.makedirs(energy_save, exist_ok=True)

    # Load test data
    test_dataset = BatchDataReader.CubeDataset(testroot, data_size,
                                               input_filename, label_filename, energy_filename)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

    net.eval()
    pbar = tqdm(enumerate(BackgroundGenerator(test_loader)), total=len(test_loader))

    # Initialize metric storage - added INF and COR metrics
    metrics_sum = {k: 0.0 for k in ['Dice', 'Accuracy', 'Sensitivity', 'Specificity', 'clDice', 'HD95', 'INF', 'COR']}
    metrics_count = {k: 0 for k in ['Dice', 'Accuracy', 'Sensitivity', 'Specificity', 'clDice', 'HD95', 'INF', 'COR']}

    sample_count = 0
    valid_sample_count = 0  # Number of samples with valid metrics

    for itr, (test_images, test_labels, energy_labels, cubename) in pbar:
        test_images = test_images.to(device=device, dtype=torch.float32)
        test_labels = test_labels.permute(0, 3, 1, 2).to(device=device, dtype=torch.float32)

        with torch.no_grad():
            # pred3 = net(test_images)  # shape: [1, 2, H, W]
            pred_energy, pred3 = net(test_images)

        pred = torch.sigmoid(pred3)  # Probability map
        pred_np = pred.squeeze(0).cpu().numpy()  # [2, H, W], float probabilities
        label_np = test_labels.squeeze(0).cpu().numpy()  # [2, H, W]

        # Get energy field predictions
        energy_np = pred_energy.squeeze(0).cpu().numpy()  # [en_channel, H, W]

        # Convert to binary 0/1
        pred_bin_np = (pred_np > threshold).astype(np.uint8)
        gt_bin_np = (label_np > 0.5).astype(np.uint8)

        # Calculate average metrics for artery and vein in current sample
        sample_metrics = []

        for c in range(2):  # 0: artery, 1: vein
            if np.any(gt_bin_np[c]) or np.any(
                    pred_bin_np[c]):  # Calculate only if ground truth or prediction has values
                metrics = utils.compute_metrics(pred_bin_np[c], gt_bin_np[c])
                sample_metrics.append(metrics)

        # If valid metrics exist, calculate average and accumulate
        if sample_metrics:
            valid_sample_count += 1
            for metric in metrics_sum.keys():
                values = [m[metric] for m in sample_metrics if not np.isnan(m[metric])]
                if values:  # If valid values exist
                    avg_value = np.mean(values)
                    metrics_sum[metric] += avg_value
                    metrics_count[metric] += 1

        # Save result images (following reference code format)
        try:
            base_name = os.path.splitext(cubename[0])[0]

            # Get original image dimensions (from the first label file)
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

            # 1. Binarization processing
            artery_mask = (pred_np[0] >= threshold).astype(np.uint8) * 255  # Artery
            vein_mask = (pred_np[1] >= threshold).astype(np.uint8) * 255  # Vein

            # 2. Resize back to original dimensions
            artery_mask_resized = cv2.resize(artery_mask, original_size, interpolation=cv2.INTER_NEAREST)
            vein_mask_resized = cv2.resize(vein_mask, original_size, interpolation=cv2.INTER_NEAREST)

            # 3. Save artery and vein segmentation results
            artery_path = os.path.join(artery_save, f"{base_name}.png")
            vein_path = os.path.join(vein_save, f"{base_name}.png")

            cv2.imwrite(artery_path, artery_mask_resized)
            cv2.imwrite(vein_path, vein_mask_resized)

            # 4. Resize probability maps to create RGB composite image
            pred_artery_resized = cv2.resize(pred_np[0], original_size, interpolation=cv2.INTER_LINEAR)
            pred_vein_resized = cv2.resize(pred_np[1], original_size, interpolation=cv2.INTER_LINEAR)

            # 5. Create RGB composite image
            BGR = np.zeros((original_size[1], original_size[0], 3), dtype=np.uint8)
            BGR[:, :, 2] = (pred_artery_resized * 255).astype(np.uint8)  # R - Artery
            BGR[:, :, 1] = (pred_vein_resized * 255).astype(np.uint8)  # G - Vein

            # 6. Save visualization results
            visual_path = os.path.join(visual_save, cubename[0])
            cv2.imwrite(visual_path, BGR)

            # 7. Save energy field probability maps
            if energy_np is not None:
                energy_channels = energy_np.shape[0]
                for ch in range(energy_channels):
                    energy_channel = energy_np[ch]
                    energy_resized = cv2.resize(energy_channel, original_size, interpolation=cv2.INTER_LINEAR)
                    # Convert probability map to 0-255 range
                    energy_prob = (energy_resized * 255).astype(np.uint8)
                    energy_path = os.path.join(energy_save, f"{base_name}_ch{ch}.png")
                    cv2.imwrite(energy_path, energy_prob)

        except Exception as e:
            logging.warning(f"Error saving image {cubename[0]}: {e}")

        sample_count += 1
        pbar.set_description(f"Testing... Samples: {sample_count}")

    # Calculate average metrics
    avg_metrics = {}
    for metric in metrics_sum.keys():
        if metrics_count[metric] > 0:
            avg_metrics[metric] = metrics_sum[metric] / metrics_count[metric]
        else:
            avg_metrics[metric] = np.nan

    # Print results - added formatted output for INF and COR
    print("\n" + "=" * 50)
    print(f"Testing completed! Total samples: {sample_count}, Valid samples: {valid_sample_count}")
    for metric, value in avg_metrics.items():
        if metric == 'HD95':
            if not np.isnan(value):
                print(f"{metric}: {value:.2f}")
            else:
                print(f"{metric}: N/A")
        elif metric in ['INF', 'COR']:
            if not np.isnan(value):
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: N/A")
        else:
            if not np.isnan(value):
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: N/A")
    print("=" * 50)
    return avg_metrics


if __name__ == '__main__':
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Load configuration
    opt = TestOptions().parse()

    # Setup GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')

    # Load network
    en_channel = len(opt.energy_filename)
    net = MAUNet2(in_channels=opt.in_channels, n_classes=opt.n_classes, channels=opt.channels, channels1st=en_channel)

    # Load trained model
    bestmodelpath = os.path.join(opt.saveroot, 'best_model',
                                 natsort.natsorted(os.listdir(os.path.join(opt.saveroot, 'best_model')))[-1])
    restore_path = os.path.join(opt.saveroot, 'best_model',
                                natsort.natsorted(os.listdir(os.path.join(opt.saveroot, 'best_model')))[-1]) + '/' + \
                   os.listdir(bestmodelpath)[0]
    net.load_state_dict(torch.load(restore_path, map_location=device))
    net.to(device=device)

    # Define list of datasets to test (fixed: added saveroot for all datasets)
    datasets = [
        {
            'name': 'OCTA500',
            'testroot': 'G:\\AV Segmentation\\Data\\OCTA500\\test',
            'saveroot': os.path.join(opt.saveroot, 'OCTA500_TEST')
        },
    ]

    try:
        # Save evaluation results for all datasets
        all_results = {}

        for dataset in datasets:
            logging.info(f'\nTesting dataset: {dataset["name"]}')

            # Create save directory
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

        # Save summary results to CSV
        try:
            summary_data = []
            for dataset_name, metrics in all_results.items():
                row = {'dataset': dataset_name}
                for metric_name, metric_value in metrics.items():
                    row[metric_name] = metric_value
                summary_data.append(row)

            df_summary = pd.DataFrame(summary_data)
            summary_path = os.path.join(opt.saveroot, 'all_datasets_summary.csv')
            df_summary.to_csv(summary_path, index=False, float_format='%.4f')
            logging.info(f"Summary results for all datasets saved to: {summary_path}")

        except Exception as e:
            logging.error(f"Error saving summary results: {e}")

        print("\nAll tests completed!")

    except KeyboardInterrupt:
        logging.info('Testing interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
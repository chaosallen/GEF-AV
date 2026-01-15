import torch
import torch.nn as nn
import numpy as np
import logging
import sys
import os
from models.MAUNet import MAUNet
import utils
import shutil
import natsort
from options.train_options import TrainOptions
import BatchDataReader
import scipy.misc as misc
from tqdm import tqdm
from prefetch_generator import BackgroundGenerator
from torchsummary import summary


def train_net(net, device):
    # Create directories for checkpoints and best model
    model_save_path = os.path.join(opt.saveroot, 'checkpoints')
    best_model_save_path = os.path.join(opt.saveroot, 'best_model')
    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(best_model_save_path, exist_ok=True)

    # === Restore best_valid_miou from previous best_model if exists ===
    existing = os.listdir(best_model_save_path)
    if len(existing) > 0:
        best_valid_miou = max([float(x) for x in existing])
    else:
        best_valid_miou = 0.0

    # === Load Dataset ===
    train_dataset = BatchDataReader.CubeDataset(opt.trainroot, opt.data_size,
                                                opt.input_filename, opt.label_filename, opt.energy_filename,
                                                is_dataaug=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)

    valid_dataset = BatchDataReader.CubeDataset(opt.testroot, opt.data_size,
                                                opt.input_filename, opt.label_filename, opt.energy_filename,
                                                is_dataaug=False)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)

    # === Setup Optimizer ===
    if opt.optimizer == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), opt.lr, momentum=0.9, weight_decay=1e-6)
    elif opt.optimizer == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), opt.lr, betas=(0.9, 0.99))
    elif opt.optimizer == 'RMS':
        optimizer = torch.optim.RMSprop(net.parameters(), opt.lr, weight_decay=1e-8)

    # === Loss Functions ===
    Loss_BCE = nn.BCEWithLogitsLoss()  # For segmentation, uses logits internally
    Loss_L1 = nn.L1Loss()  # For energy map regression
    switch_interval = 30  # Interval for alternating training phases
    phase_cycle = 2  # Number of training phases (segmentation, energy)

    # === Training Loop ===
    for epoch in range(1, opt.num_epochs + 1):
        net.train()
        running_loss = 0
        pbar = tqdm(enumerate(BackgroundGenerator(train_loader)), total=len(train_loader), desc=f'Epoch {epoch}')

        for itr, (images, gt_seg, gt_energy, name) in pbar:
            images = images.to(device=device, dtype=torch.float32)
            gt_seg = gt_seg.to(device=device, dtype=torch.float32).permute(0, 3, 1, 2).contiguous()
            gt_energy = gt_energy.to(device=device, dtype=torch.float32).permute(0, 3, 1, 2).contiguous()

            # Forward pass: network outputs two tensors
            pred_energy, pred_seg_logits = net(images)

            # Compute segmentation loss (BCEWithLogitsLoss expects logits)
            loss_seg = Loss_BCE(pred_seg_logits, gt_seg)

            # Compute L1 loss for energy estimation
            loss_energy = Loss_L1(pred_energy, gt_energy)

            # === Determine training phase ===
            phase_id = (itr // switch_interval) % phase_cycle
            if phase_id == 0:  # segmentation phase
                loss = loss_seg
            elif phase_id == 1:  # energy estimation phase
                loss = loss_energy

            # Backpropagation and optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({'Loss': f'{running_loss / (itr + 1):.4f}'})

        # === Validation ===
        with torch.no_grad():
            # Save current epoch checkpoint
            torch.save(net.state_dict(), os.path.join(model_save_path, f'{epoch}.pth'))
            logging.info(f'Checkpoint {epoch} saved!')

            val_num = 0
            val_miou_sum = 0
            net.eval()
            pbar = tqdm(enumerate(BackgroundGenerator(valid_loader)), total=len(valid_loader), desc='Validating')

            for itr, (test_images, test_annotations, test_energy, cubename) in pbar:
                test_images = test_images.to(device=device, dtype=torch.float32)
                test_annotations = test_annotations.permute(0, 3, 1, 2).to(device=device, dtype=torch.float32)

                pred1, pred2 = net(test_images)
                pred = torch.sigmoid(pred2)
                pred = (pred > 0.5).float()

                val_miou_sum += utils.cal_miou_multilabel(pred, test_annotations)
                val_num += 1

            val_miou = val_miou_sum / val_num
            print(f"Step: {epoch}, Valid mIoU: {val_miou:.4f}")

            # === Save best model if improved ===
            if val_miou > best_valid_miou:
                temp = '{:.6f}'.format(val_miou)
                best_path = os.path.join(best_model_save_path, temp)
                os.makedirs(best_path, exist_ok=True)

                best_model_file = os.path.join(model_save_path, f'{epoch}.pth')
                shutil.copy(best_model_file, os.path.join(best_path, f'{epoch}.pth'))

                # Keep at most 3 best model folders
                model_folders = natsort.natsorted(os.listdir(best_model_save_path))
                if len(model_folders) > 3:
                    shutil.rmtree(os.path.join(best_model_save_path, model_folders[0]))

                best_valid_miou = val_miou


if __name__ == '__main__':
    # Set logging configuration
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Load training options
    opt = TrainOptions().parse()

    # Set GPU device
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    en_channel = len(opt.energy_filename)

    # Initialize network
    net = MAUNet(in_channels=opt.in_channels, n_classes=opt.n_classes, channels=opt.channels,
                 channels1st=en_channel)
    net = net.to(device)

    # Load pretrained model if specified
    if opt.load:
        net.load_state_dict(torch.load(opt.load, map_location=device))
        logging.info(f'Model loaded from {opt.load}')

    try:
        train_net(net=net, device=device)
    except KeyboardInterrupt:
        # Save model on interruption
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

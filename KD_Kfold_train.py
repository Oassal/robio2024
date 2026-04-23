import torch
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from sklearn.model_selection import KFold
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
import os
from pathlib import Path
from models.models_student import VAE, ResNet_VAE
from models.models_teacher import Teacher
from utils.utils_new import inference_VAE, metric, metric_clustersCenters
import logging
from models.SegModel import DeepLabV3
from ultralytics import YOLO
import random
import argparse
from utils.utils_new import feature_points_finding
import torch.nn.functional as F
from trainers.kfold_functions import create_crossfold_splits, save_splits_to_yaml
from utils.utils_new import create_list_directories
from dataset_kd import VAEDataset_NoFlags as KD_Dataset
from tqdm import tqdm 
from trainers.heatmap_analysis import extract_points_batch as extract_points
from trainers.heatmap_analysis import evaluate_heatmaps 

from trainers.trainers import train_epoch
### Seeding to ensure repetitivity ###
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
######################################
parser = argparse.ArgumentParser(description='Train a VAE model with k-fold cross-validation.')
parser.add_argument('--data_path', type=str, default =r'Dataset', required=False, help='Path to the dataset')
parser.add_argument('--num_folds', type=int, default=5, help='Number of folds for cross-validation')
parser.add_argument('--num_epochs', type=int, default=2, help='Number of epochs for training per fold')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for the optimizer')
parser.add_argument('--model_save_path', type=str, default='models_kfold', help='Path to save the trained models')
parser.add_argument('--log_dir', type=str, default='runs_kfold', help='Directory for TensorBoard logs')
parser.add_argument('--segmentation_model_path', type=str, default = r'fold0_model.pth', required=False, help='Path to the pretrained segmentation model')
parser.add_argument('--segmentation_model_type', type=str, default='deeplab', choices=['deeplab', 'yolo'], help='Type of segmentation model to use')
parser.add_argument('--teacher_model_path', type=str, required=False, help='Path to the pretrained teacher model')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for training (cuda or cpu)')
parser.add_argument('--output_zone_shape', type=str, default='DoubleChannel_AutoGaussian', help='Type of output zone shape to use')
parser.add_argument('--load_checkpoint', action='store_true', help='Flag to load from checkpoint if available')
parser.add_argument('--checkpoint_path', type=str, default='checkpoint.pth', help='Path to the checkpoint file')
parser.add_argument('--threshold', type=float, default=2.0, help='Threshold for angles filtering')
parser.add_argument('--student_model', type=str, default='VAE', help='Type of student model to use')
parser.add_argument('--student_backbone', type=str, default='resnet34', help='Backbone for the student model if applicable')
parser.add_argument('--nb_clusters', type=int, default=4, help='Number of clusters for heatmaps generation algorithm')
parser.add_argument('--randomize_cluster', action='store_true', help='Flag to randomize the clusters centers')
parser.add_argument('--ckpt_freq', type=int, default=10, help='Frequency (in epochs) to save model checkpoints')
parser.add_argument('--temp_dir', type=str, default='temp', help='Directory to save temporary files during training')
args = parser.parse_args()

### Parameters from argparse ###
DATA_PATH = args.data_path
# DATA_PATH = r'C:\Users\oel_assa\OneDrive - Alstom\Documents\Dataset\Changing wire feed rate'
NUM_FOLDS = args.num_folds
NUM_EPOCHS = args.num_epochs
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate
MODEL_SAVE_PATH = args.model_save_path
LOG_DIR = args.log_dir
SEGMENTATION_MODEL_PATH = args.segmentation_model_path
# SEGMENTATION_MODEL_PATH = r'C:\Users\oel_assa\OneDrive - Alstom\Documents\Codes\Python codes\ROBIO\fold0_model.pth'
TEACHER_MODEL_PATH = args.teacher_model_path
DEVICE = args.device
TEMP_DIR = args.temp_dir

### Create necessary directories if they don't exist ###
Path.mkdir(Path(MODEL_SAVE_PATH), parents=True, exist_ok=True)
Path.mkdir(Path(LOG_DIR), parents=True, exist_ok=True)
Path.mkdir(Path(TEMP_DIR), parents=True, exist_ok=True)

### TensorBoard Writer ###
writer = SummaryWriter(LOG_DIR)
###########################

''' Models intialization '''
# Teacher Model
if args.segmentation_model_type == 'deeplab':
    seg_model = DeepLabV3(3)
    seg_model.to(DEVICE)
    seg_model.load_state_dict(torch.load(SEGMENTATION_MODEL_PATH))
    seg_model.eval()
elif args.segmentation_model_type == 'yolo':
    seg_model = YOLO(SEGMENTATION_MODEL_PATH)
    seg_model.to(DEVICE)
    seg_model.eval()

# Initialize the teacher model with the segmentation model and heatmaps finder
teacher_model = Teacher(segmentation_model=seg_model,
                        seg_model_name=args.segmentation_model_type,
                        heatmaps_finder=feature_points_finding,
                        )
teacher_model.to(DEVICE)
teacher_model.eval()
# Initialize student model
student_model = ResNet_VAE(model_name=args.student_backbone, NB_OUTPUT_CHANNELS=2).to(DEVICE)

# Initialize loss functions
mse_loss = F.mse_loss

# optimizer will be created per-fold after student model is reinitialized


video_dirs = create_list_directories(DATA_PATH)
video_dirs_out = [path['out'] for path in video_dirs]
kfolds = create_crossfold_splits(video_dirs_out, window_size=1)

save_splits_to_yaml(kfolds, 'kd_kfold_splits.yaml')
if __name__== "__main__":
    # freeze_support()
    for fold, dirFold in enumerate(kfolds):
        print(f'Starting fold {fold + 1}/{NUM_FOLDS}')
        logging.info(f'Starting fold {fold + 1}/{NUM_FOLDS}')
        
        ## Create datasets and dataloaders for the current fold
        train_dirs = dirFold['train']
        val_dirs = dirFold['val']
        
        train_dataset = ConcatDataset([KD_Dataset(
            miniDataset, 
            seg_model,
            model_seg_type=args.segmentation_model_type,
            augment=True,
            normalize=True,
            outputZoneShape=args.output_zone_shape,
            thresh=args.threshold,
            n_clusters=args.nb_clusters,
            # randomize_cluster=args.randomize_cluster
        ) for miniDataset in train_dirs])

        val_dataset = ConcatDataset([KD_Dataset(
            miniDataset, 
            seg_model,
            model_seg_type=args.segmentation_model_type,
            augment=False,
            normalize=True,
            outputZoneShape=args.output_zone_shape,
            thresh=args.threshold,
            n_clusters=args.nb_clusters,
            # randomize_cluster=False
        ) for miniDataset in val_dirs])

        # Dataloaders
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        # Reset model weights before training each fold
        student_model = ResNet_VAE(model_name=args.student_backbone, NB_OUTPUT_CHANNELS=2).to(DEVICE)
        # recreate optimizer so it points to the new model parameters
        optimizer = optim.Adam(student_model.parameters(), lr=LEARNING_RATE)

        # Load from checkpoint if specified
        start_epoch = 0
        if args.load_checkpoint and os.path.exists(args.checkpoint_path):
            checkpoint = torch.load(args.checkpoint_path)
            student_model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f'Loaded checkpoint from {args.checkpoint_path}, starting from epoch {start_epoch}')
            logging.info(f'Loaded checkpoint from {args.checkpoint_path}, starting from epoch {start_epoch}')
        
        best_val_loss = float('inf')
        # Select a random sample from the validation set for visualization
        random_val_sample = random.choice(val_dataset)

        # Add the random sample to TensorBoard before training
        img, label = random_val_sample
        img = img.unsqueeze(0).to(DEVICE)  # Add batch dimension and move to device
        teacher_output = teacher_model(img).squeeze(0).cpu()
        print(teacher_output.shape)
        teacher_output_1 = teacher_output[0].squeeze(0).cpu()
        teacher_output_2 = teacher_output[1].squeeze(0).cpu()
        writer.add_image(f'Fold_{fold+1}/Random_Sample/Teacher_Output_Before_Training_1', teacher_output_1, 0, dataformats='HW')
        writer.add_image(f'Fold_{fold+1}/Random_Sample/Teacher_Output_Before_Training_2', teacher_output_2, 0, dataformats='HW')
        writer.flush()
        for epoch in tqdm(range(start_epoch, NUM_EPOCHS)):
            break
            student_model.train()
            train_loss = 0.0
            # train_loss = train_epoch(teacher_model, 
            #                          student_model, 
            #                          DEVICE, 
            #                          train_loader, 
            #                          optimizer, 
            #                          criterion=mse_loss, 
            #                          alpha=1.0, 
            #                          epoch=epoch,
            #                          NUM_EPOCHS=NUM_EPOCHS)
                
            train_loss /= len(train_loader.dataset)
            writer.add_scalar(f'Fold_{fold+1}/Train/Loss', train_loss, epoch)
            writer.flush()
            print(f'Fold {fold + 1}, Epoch {epoch + 1}, Train Loss: {train_loss:.4f}')
            logging.info(f'Fold {fold + 1}, Epoch {epoch + 1}, Train Loss: {train_loss:.4f}')

            if epoch % args.ckpt_freq == 0 or epoch == NUM_EPOCHS - 1:
                checkpoint_path = os.path.join(MODEL_SAVE_PATH, f'fold{fold}_epoch{epoch}_checkpoint_kd.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': student_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, checkpoint_path)
                print(f'Saved checkpoint to {checkpoint_path}')
                logging.info(f'Saved checkpoint to {checkpoint_path}')
            
            # Test on the random validation sample and log to TensorBoard
            student_model.eval()
            with torch.no_grad():
                img, label = random_val_sample
                img = img.unsqueeze(0).to(DEVICE)  # Add batch dimension and move to device
                student_output, _, _, _ = student_model(img)
                teacher_output = teacher_model(img)
                print(student_output.shape)
                student_output = student_output.squeeze(0).cpu()
                teacher_output = teacher_output.squeeze(0).cpu()
                student_output_img_1 = student_output[0].numpy()
                student_output_img_2 = student_output[1].numpy()
                teacher_output_img_1 = teacher_output[0].numpy()
                teacher_output_img_2 = teacher_output[1].numpy()
                # Log the student and teacher outputs as images in TensorBoard, shape (2, H, W -> H, W)
                writer.add_image(f'Fold_{fold+1}/Random_Sample/Student_Output_1', student_output[0].squeeze(0).cpu(), epoch, dataformats='HW')
                writer.add_image(f'Fold_{fold+1}/Random_Sample/Student_Output_2', student_output[1].squeeze(0).cpu(), epoch, dataformats='HW')
                writer.add_image(f'Fold_{fold+1}/Random_Sample/Teacher_Output_1', teacher_output[0].squeeze(0).cpu(), epoch, dataformats='HW')
                writer.add_image(f'Fold_{fold+1}/Random_Sample/Teacher_Output_2', teacher_output[1].squeeze(0).cpu(), epoch, dataformats='HW')
                writer.flush()
            # Save the result to the temporary directory for later analysis
            temp_output_path = os.path.join(TEMP_DIR, f'fold{fold}_epoch{epoch}_sample_output.pt')
            torch.save({
                'student_output': student_output.cpu(),
                'teacher_output': teacher_output.cpu(),
                'label': label
            }, temp_output_path)

            # Save as png for visualization
            student_output_img_1 = student_output[0].squeeze(0).cpu().numpy()
            student_output_img_2 = student_output[1].squeeze(0).cpu().numpy()
            teacher_output_img_1 = teacher_output[0].squeeze(0).cpu().numpy()
            teacher_output_img_2 = teacher_output[1].squeeze(0).cpu().numpy()
            cv2.imwrite(os.path.join(TEMP_DIR, f'fold{fold}_epoch{epoch}_student_output_1.png'), student_output_img_1 * 255)
            cv2.imwrite(os.path.join(TEMP_DIR, f'fold{fold}_epoch{epoch}_student_output_2.png'), student_output_img_2 * 255)
            cv2.imwrite(os.path.join(TEMP_DIR, f'fold{fold}_epoch{epoch}_teacher_output_1.png'), teacher_output_img_1 * 255)
            cv2.imwrite(os.path.join(TEMP_DIR, f'fold{fold}_epoch{epoch}_teacher_output_2.png'), teacher_output_img_2 * 255)

        # Validation after training each fold
        student_model.eval()
        val_loss = 0.0

        all_distances = {
            "gt-manual": [],
            "teacher-manual": [],
            "gt-teacher": []
        }
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Validation Fold {fold + 1}"):
                images = images.to(DEVICE)
                # TODO transform these to tensors and move them to the device
                manual_gt_points = labels['manual_gt']
                teacher_points = labels['teacher_points']

                # single forward pass for student
                student_output, decoded, z_mean, z_log_var = student_model(images)

                # Teacher output
                teacher_output = teacher_model(images)
                kd_loss = mse_loss(student_output, teacher_output)

                # Loss calculation
                kl_div = -0.5 * torch.sum(1 + z_log_var - z_mean**2 - torch.exp(z_log_var), dim=1)
                loss = kd_loss + kl_div.mean()
                val_loss += loss.item() * images.size(0)

                batch_med = evaluate_heatmaps(
                    teacher_output,
                    manual_gt_points,
                    teacher_points,
                    nb_zones=args.nb_clusters,
                    threshold=0.3,
                    retrieval_type="clustering"
                )
                all_distances["gt-manual"].extend(batch_med["gt-manual"])
                all_distances["teacher-manual"].extend(batch_med["teacher-manual"])
                all_distances["gt-teacher"].extend(batch_med["gt-teacher"])

        print(f'Validation metrics for fold {fold + 1}:')
        for key, values in all_distances.items():
            print(f'  {key}: {np.mean(values):.4f}')
            writer.add_scalar(f'Fold_{fold+1}/Validation/{key}', np.mean(values), epoch)
            writer.flush()
        print(f'--- End of fold {fold + 1} ---')

writer.close()
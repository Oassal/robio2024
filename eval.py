from dataset_kd import VAEDataset_NoFlags as KD_Dataset
from models.models_teacher import Teacher
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from trainers.heatmap_analysis import calculate_distance_lists
import numpy as np
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt
import argparse
import cv2
import torch


def evaluate(teacher_model, student_model, device, dataset, epoch = 0, writer=None, fold=0, write_video=False,
             nb_zones=4, threshold=0.3, retrieval_type="clustering"):
    '''
    Evaluate the student model against the teacher model and the ground truth points using the provided validation loader.
    This function calculates the distances between the predicted heatmap points from the student model, the ground truth points, and the teacher points. It logs the average distances to TensorBoard if a writer is provided.
    Args:
        teacher_model: The pre-trained teacher model used for knowledge distillation.
        student_model: The student model being evaluated.
        device: The device (CPU or GPU) on which the models and data are located.
        dataset: The dataset for evaluation, providing batches of images, ground truth points, and teacher points.
        epoch: The current epoch number, used for logging purposes.
        writer: An optional SummaryWriter for logging metrics to TensorBoard.
        fold: The current fold number in k-fold cross-validation, used for logging purposes.
        write_video: A boolean flag indicating whether to write the evaluation results to a video file.
    '''
    teacher_model.eval()
    student_model.eval()
    print(f'--- Evaluating fold {fold + 1} ---')
    print(f'Number of samples in evaluation dataset: {len(dataset)}, video writing: {write_video}')
    if write_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(f'evaluation_fold{fold+1}_epoch{epoch}.mp4', fourcc, 10.0, (1280, 720))

    all_distances = {
        "gt-manual": [],
        "teacher-manual": [],
        "gt-teacher": [],
        "perPoint-student-teacher": [],
        "perPoint-student-gt": []
    }
    with torch.no_grad():
        for img, mask, pts, gt_points in dataset:
            images = img.to(device)
            gt_points = gt_points.to(device)
            teacher_points = pts.to(device)

            student_output, _, _, _ = student_model(images)
            teacher_output = teacher_model(images)

            # Convert outputs to CPU numpy arrays for distance calculation
            student_output = student_output.cpu().numpy()
            teacher_output = teacher_output.cpu().numpy()
            gt_points = gt_points.cpu().numpy()
            teacher_points = teacher_points.cpu().numpy()

            # Calculate distances between predicted heatmap points and ground truth/teacher points
            distances = calculate_distance_lists(student_output, gt_points, teacher_points, nb_zones=5)
            all_distances["gt-manual"].extend(distances["gt-manual"])
            all_distances["teacher-manual"].extend(distances["teacher-manual"])
            all_distances["gt-teacher"].extend(distances["gt-teacher"])

            # Calculate the per point distance
            # Student/GT
            dist_matrix_student = cdist(student_output, gt_points, metric='euclidean')
            min_distance_student = np.min(dist_matrix_student, axis=1)
            all_distances["perPoint-student-gt"].extend(min_distance_student)

            # Student/Teacher
            dist_matrix_teacher = cdist(student_output, teacher_points, metric='euclidean')
            min_distance_teacher = np.min(dist_matrix_teacher, axis=1)
            all_distances["perPoint-student-teacher"].extend(min_distance_teacher)


            # Optionally, write the evaluation results to a video file
            if write_video:
                # Plot the original image, the student output heatmap, the teacher output heatmap, and the ground truth points
                fig, axs = plt.subplots(1, 4, figsize=(20, 5))
                axs[0].imshow(img.permute(1, 2, 0).cpu())
                axs[0].set_title('Original Image')
                axs[1].imshow(student_output[0], cmap='hot')
                axs[1].set_title('Student Output Heatmap')
                axs[2].imshow(teacher_output[0], cmap='hot')
                axs[2].set_title('Teacher Output Heatmap')
                axs[3].imshow(gt_points[0], cmap='hot')
                axs[3].set_title('Ground Truth Points')

                # Add the predicted points from the student model and the teacher model to the original image
                axs[0].scatter(student_output[:, 0], student_output[:, 1], c='blue', label='Student Predicted Points')
                axs[0].scatter(teacher_points[:, 0], teacher_points[:, 1], c='red', label='Teacher Predicted Points')
                axs[0].scatter(gt_points[:, 0], gt_points[:, 1], c='green', label='Ground Truth Points')
                axs[0].legend()

                # Add the predicted points from the student model and the teacher model to the respective heatmaps
                axs[1].scatter(student_output[:, 0], student_output[:, 1], c='blue', label='Student Predicted Points')
                axs[1].legend()
                axs[2].scatter(teacher_points[:, 0], teacher_points[:, 1], c='red', label='Teacher Predicted Points')
                axs[2].legend()
                axs[3].scatter(gt_points[:, 0], gt_points[:, 1], c='green', label='Ground Truth Points')
                axs[3].legend()
                
                plt.tight_layout()
                # Save the figure to a video file (e.g., using OpenCV)
                cv2.imwrite(f'temp/evaluation_fold{fold+1}_epoch{epoch}_sample.png', cv2.cvtColor(np.array(fig.canvas.renderer.buffer_rgba()), cv2.COLOR_RGBA2BGR))
                plt.close(fig)
                frame = cv2.imread(f'temp/evaluation_fold{fold+1}_epoch{epoch}_sample.png')
                video_writer.write(frame)
        if write_video:
            video_writer.release()
            print(f'Video saved as evaluation_fold{fold+1}_epoch{epoch}.mp4')
    # Log the average distances to TensorBoard
    if writer is not None:
        for key, values in all_distances.items():
            writer.add_scalar(f'Fold_{fold+1}/Validation/{key}', np.mean(values), epoch)
        writer.flush()

    return all_distances

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate the student model against the teacher model and ground truth points.')
    parser.add_argument('--teacher_model_path', type=str, required=True, help='Path to the pre-trained teacher model.')
    parser.add_argument('--student_model_path', type=str, required=True, help='Path to the trained student model to be evaluated.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the evaluation dataset.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the evaluation on (e.g., "cuda" or "cpu").')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory to save TensorBoard logs.')
    parser.add_argument('--epoch', type=int, default=0, help='Epoch number for logging purposes.')
    parser.add_argument('--fold', type=int, default=0, help='Fold number for logging purposes.')
    parser.add_argument('--write_video', action='store_true', help='Whether to write the evaluation results to a video file.')

    args = parser.parse_args()

    # Load teacher and student models
    teacher_model = Teacher()  # Initialize your teacher model architecture here
    teacher_model.load_state_dict(torch.load(args.teacher_model_path))
    teacher_model.to(args.device)
    teacher_model.eval()

    student_model = Teacher()  # Initialize your student model architecture here (should match the one used during training)
    student_model.load_state_dict(torch.load(args.student_model_path))
    student_model.to(args.device)
    student_model.eval()

    # Load evaluation dataset
    eval_dataset = KD_Dataset(args.data_path)  # Initialize your dataset class here

    # Create a SummaryWriter for TensorBoard logging
    writer = SummaryWriter(log_dir=args.log_dir)

    # Evaluate the models
    evaluate(teacher_model, student_model, args.device, eval_dataset, epoch=args.epoch, writer=writer, fold=args.fold, write_video=args.write_video)

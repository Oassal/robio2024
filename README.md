# ROBIO2024
This repo contains the implementation code of the paper : Shape estimation of Viscoelastic fluid materials an application for weld pool.

We present a novel approach based on knowledge destillation to analyse the shape of weld pool, through RGB images.

The overall framework is illustrated as follows:
![image info](resources\framework.png)

![image info](resources\kd.png)
# ROBIO — VAE Knowledge Distillation

**Project**: training and evaluating VAE-based student models using a pre-trained teacher (DeepLabV3 / YOLO) with k-fold cross-validation and automatic heatmap generation.

**Summary**
- Implements ResNet-based VAE student models and a Teacher wrapper that extracts heatmaps from a segmentation model and converts them to probabilistic feature maps.
- Provides k-fold training (`KD_Kfold_train.py`), dataset helpers, heatmap evaluation, and utilities for producing visualizations and TensorBoard logs.

**Requirements**
- Python 3.8+ (tested on Anaconda env)
- PyTorch + torchvision compatible with your CUDA or CPU setup
- Key packages: `torch`, `torchvision`, `albumentations`, `opencv-python`, `numpy`, `scikit-learn`, `scipy`, `matplotlib`, `tensorboard`, `ultralytics` (if using YOLO)

Install (example):
```
pip install torch torchvision albumentations opencv-python numpy scikit-learn scipy matplotlib tensorboard ultralytics
```

**Quickstart**
1. Prepare your dataset and segmentation model checkpoint(s). Update paths or pass them to the training script.
2. Run k-fold training (example):
```
python KD_Kfold_train.py \
    --data_path "path/to/dataset" \
    --segmentation_model_path fold0_model.pth \
    --log_dir runs_kfold
```
3. Start TensorBoard pointing to the absolute log directory:
```
tensorboard --logdir "full/path/to/robio2024/runs_kfold" --reload_interval 5
```
4. Evaluation:
```
python eval.py \
  --teacher_model_path "C:\full\path\to\teacher_checkpoint.pth" \
  --student_model_path "C:\full\path\to\student_checkpoint.pth" \
  --data_path "C:\full\path\to\eval_dataset" \
  --device cpu \
  --log_dir "C:\full\path\to\robio2024\runs_eval" \
  --epoch 10 \
  --fold 0 \
  --write_video
```

**TensorBoard notes & troubleshooting**
- Use an absolute `--logdir` path to avoid ambiguity.
- If your project is stored on OneDrive, sync/locking can delay writes; consider using a local non-synced folder for `--logdir`.
- The code flushes and closes the `SummaryWriter`, but if you still don't see scalars/images, restart TensorBoard and ensure the correct folder is selected.

**Files of interest**
- `KD_Kfold_train.py`: main training script with TensorBoard logging.
- `models.py` / `models/models_student.py` / `models/models_teacher.py`: model definitions.
- `utils_new.py` (`utils/utils_new.py`): utility functions (heatmap generation, inference helpers).
- `dataset_kd.py`: dataset class used for KD.
- `trainers/`: training and evaluation helper functions.

**Tips**
- Run a short test (1 epoch, small batch) to verify logging before starting full experiments.
- Keep checkpoints and logs outside OneDrive to avoid sync issues.

**Citation**
If you use this work, please cite:
```
@article{assal2024leveraging,
  title={Leveraging Foundation Models To learn the shape of semi-fluid deformable objects},
  author={Assal, Omar El and Mateo, Carlos M and Ciron, Sebastien and Fofi, David},
  journal={arXiv preprint arXiv:2411.16802},
  year={2024}
},
@inproceedings{el2024shape,
  title={Shape Estimation of Viscoelastic Fluid Materials: An Application for Weld Pool},
  author={El Assal, Omar and Mateo, Carlos M and Ciron, Sebastien and Fofi, David},
  booktitle={2024 IEEE International Conference on Robotics and Biomimetics (ROBIO)},
  pages={426--433},
  year={2024},
  organization={IEEE}
}
```
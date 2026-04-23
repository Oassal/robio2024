import torch

from scipy.ndimage.measurements import center_of_mass
from skimage import measure
from sklearn.cluster import KMeans
import numpy as np
from scipy.spatial.distance import cdist

def extract_points_image(heatmap: torch.Tensor, nb_zones: int, threshold: float = 0.3, retrieval_type: str | None = "clustering") -> list:
    centers = []
    binary = heatmap > threshold
    binary = binary[0].cpu().numpy()  # Assuming heatmap has shape (1, H, W)
    if retrieval_type == "clustering":
        #TODO implement clustering based point extraction
        x =[]
        y = []

        for i in range(binary.shape[0]):
            for j in range(binary.shape[1]):
                if binary[i,j]:
                    x.append(i)
                    y.append(j)
        points = np.array(list(zip(x,y)))
        if len(points) > 0:
            n_clusters = min(nb_zones, len(points))  # Adjust the number of clusters as needed
            kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(points)
            cluster_centers = kmeans.cluster_centers_
            centers = [(center[0], center[1]) for center in cluster_centers]
        else:
            centers = [-1,-1]

    elif retrieval_type == "max_zones":
        #TODO implement max zones based point extraction
        labeled = measure.label(binary)
        regions = measure.regionprops(labeled,intensity_image=heatmap)
        # centers = [center_of_mass(region.intensity_image) for region in regions]
        centers = [(X,Y) for region in regions for X,Y in [region.centroid]] #FIXME this one is to check
        for props in regions:
            X,Y = props.centroid
            centers.append((X,Y))
        centers = sorted(centers, key=lambda x: heatmap[int(x[0]), int(x[1])], reverse=True)[:nb_zones]

    # Implement the logic to extract points from the heatmap
    return centers


def extract_points_batch(batch_heatmaps: torch.Tensor, nb_zones: int, threshold: float = 0.3, retrieval_type: str | None = "clustering") -> list:
    all_centers = []
    for heatmap in batch_heatmaps:
        centers = extract_points_image(heatmap, nb_zones, threshold, retrieval_type)
        all_centers.append(centers)
    return all_centers


def calculate_distance(predicted_points: list, ground_truth_points: list) -> list:
    #TODO transform this function into a function that calculates the distance between two heatmaps instead

    distances = []
    for pred, gt in zip(predicted_points, ground_truth_points):
        # if pred == [-1,-1] or gt == [-1,-1]:
        #     distances.append(float('inf'))  # Assign a large distance if points are missing
        #     continue
        # dist = np.sqrt((pred[0] - gt[0]) ** 2 + (pred[1] - gt[1]) ** 2)
        # distances.append(dist)
        dist_matrix = cdist([pred], [gt], metric='euclidean')
        min_distance = np.min(dist_matrix, axis=1)
        distances.append(dist_matrix[0][0])
    return distances


def calculate_distance_lists(predicted_heatmap_points, ground_truth_points: list, teacher_points: list, nb_zones: int) -> dict:
    
    distances = {
        "gt-manual": calculate_distance(predicted_heatmap_points, ground_truth_points),
        "teacher-manual": calculate_distance(predicted_heatmap_points, teacher_points),
        "gt-teacher": calculate_distance(teacher_points, ground_truth_points)
    }
    return distances

def evaluate_heatmaps(batch_heatmaps: torch.Tensor, ground_truth_points: list, teacher_points: list, nb_zones: int, threshold: float = 0.3, retrieval_type: str | None = "clustering") -> dict:
    all_distances = {
        "gt-manual": [],
        "teacher-manual": [],
        "gt-teacher": []
    }
    for density_map, gt_pts, teacher_pts in zip(batch_heatmaps, ground_truth_points, teacher_points):
        heatmap_pts = extract_points_image(density_map, nb_zones, threshold, retrieval_type)
        gt_pts = gt_pts.cpu().numpy() if isinstance(gt_pts, torch.Tensor) else gt_pts
        teacher_pts = teacher_pts.cpu().numpy() if isinstance(teacher_pts, torch.Tensor) else teacher_pts
        distances = calculate_distance_lists(heatmap_pts, gt_pts, teacher_pts, nb_zones)
        all_distances["gt-manual"].extend(distances["gt-manual"])
        all_distances["teacher-manual"].extend(distances["teacher-manual"])
        all_distances["gt-teacher"].extend(distances["gt-teacher"])
    return all_distances
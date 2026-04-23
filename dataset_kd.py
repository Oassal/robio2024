import os
import io
from typing import Dict
import torch
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms, utils
import cv2
from pathlib import Path
import collections
import albumentations as alb
from utils.utils_new import feature_points_finding
from models.SegModel import DeepLabV3
from albumentations.pytorch import ToTensorV2
import random
import json
from albumentations.augmentations.geometric.transforms import Affine
from torchvision.transforms import ToTensor
from scipy.ndimage.measurements import center_of_mass
from skimage import measure
from utils.yolo_functions import extract_masks_with_tracking
from typing import Optional
from ultralytics import YOLO

# torch.manual_seed(0)
# random.seed(0)
# np.random.seed(0)

def _normalize_tensor(unnormalized:torch.tensor)->torch.tensor:
    '''
    A function to normalize a pytorch tensor 'A' of shape (W,H)
     according to the forumla: (a-min(A))/max(A)
    '''
    normalized = unnormalized.clone()
    normalized = normalized.view(-1)
    normalized -=normalized.min(0,keepdim = True)[0]
    if not normalized.max(0,keepdim = True)[0] == torch.tensor(0):
        normalized /=normalized.max(0,keepdim = True)[0]
    else : 
        normalized *=normalized.max(0,keepdim = True)[0]
    normalized = normalized.view(unnormalized.shape)
    return normalized


class ToTensor(alb.ImageOnlyTransform):
    def __init__(self):
        super().__init__(always_apply=True)
    
    def apply(self, img, **params):
        return transforms.ToTensor()(img)
    def get_params(self) -> Dict:
        return {}


class VAEDataset_NoFlags(Dataset):
    def __init__(self, dir, model_seg, model_seg_type,outputZoneShape = 'SingleChannel_Gaussian',
                 thresh = 1,augment=False,normalize=False, normalize_GT = True, n_clusters : Optional[int] = 4,gaussian_contour = True) -> None:
        '''
        Dataset class for the VAE training, it takes as input the path to the dataset, the segmentation model to be used for generating the feature points and some other parameters related to the output and the data augmentation. It returns a tuple of (image, teacher_output, manual_gt_points, teacher_points).
        Parameters:
        dir : path to dataset
        SegModelPath : path to segmentation model used by the teacher for generating the feature points
        model_seg_type : type of the segmentation model (deeplab or yolo)
        outputZoneShape: shape of the output (two channels or single...etc)
        Thresh : threshold for candidate normal lines finding,
        augment : apply image data augmentation or not
        normalize : normalize the final image (not each distribution)
        normalize_GT : normalize the masks while finding them
        n_clusters: the number of heatmaps to be generated, if None it is randomized between 3 and 10
        gaussian_contour: describe the contours as gaussian distribution 
        '''
        super().__init__()
        self.dir=Path(dir)
        self.outputZoneShape = outputZoneShape
        self.augment = augment
        self.imagePaths=[]
        self.segmentor = model_seg
        self.model_seg_type = model_seg_type
        self.thresh = thresh
        self.normalize = normalize
        self.n_clusters = n_clusters
        self.normalize_GT=normalize_GT
        self.gaussian_contour = gaussian_contour
        # #this output image of segmentation is taken directly, which is of size
        #1080//4 , 1440//4, so no need to resize here
        augment_transforms = [
                            #   alb.Resize(1080//4,1440//4,always_apply=True),
                              ToTensorV2()]
        assert self.dir.exists()

        self.imagesList = []

        
        for jsonPath in self.dir.rglob('*.json'):
            manual_gt_points = []
            with open(jsonPath,'r') as jsonFile:
                jsonFile = json.load(jsonFile)
                for point_coords in jsonFile['tooltips']:
                    manual_gt_points.append((point_coords['x']//4,point_coords['y']//4))
                    # manual_gt_points = np.array(manual_gt_points)
                    # FIXME Assumed that we always have 4 points, in the future make it variable
                    if len(manual_gt_points) != 4: continue
                    self.imagesList.append(str(jsonPath).replace('.json','.jpg'))

        if self.augment:
            augment_transforms=[
                alb.Resize(1080//4,1440//4, always_apply=True),
                alb.HorizontalFlip(p=0.3),
                alb.RandomBrightnessContrast(p=0.2),
                alb.VerticalFlip(p=0.3),
                Affine(translate_px=(-45,45), p=1)
            ]+augment_transforms

        self.transforms = alb.Compose(augment_transforms,additional_targets={'mask0':'mask','mask1':'mask'})
        self.res = alb.Resize(1080//4,1440//4, always_apply=True)

    def __len__(self):
        return len(self.imagesList)

    def __getitem__(self, idx:int) -> tuple:
        manual_gt_points = []
        # Read the image and the corresponding json file, extract the manual GT points, apply the segmentation model to get the feature points and return everything as a tuple
        image = cv2.imread(str(self.imagesList[idx]))
        image = cv2.resize(image, (360, 270))
        json_file = str(self.imagesList[idx]).replace('.jpg','.json')
        with open(json_file,'r') as jf:
            json_data = json.load(jf)
        
        # Extract manual GT points from JSON file
        for point_coords in json_data['tooltips']:
            manual_gt_points.append((point_coords['x']//4,point_coords['y']//4))
        manual_gt_points = np.array(manual_gt_points)

        # Check if we are using randomized number of clusters or not
        if self.n_clusters is None:
            self.n_clusters = random.randint(3, 10)
        
        # Apply the segmentation model and find the feature points using teacher model
        if self.model_seg_type == "deeplab":
            featureMap,points = feature_points_finding(self.segmentor,image, self.thresh, gaussian_contour=self.gaussian_contour,
                                                    zoneShape=self.outputZoneShape, n_clusters=self.n_clusters, normalize_GT=self.normalize_GT,idx=str(self.imagesList[idx]))
        elif self.model_seg_type == "yolo":
            yolo_masks = extract_masks_with_tracking(image,idx,self.segmentor)
            try:
                mask = yolo_masks['wire']
                # plt.imshow(mask)
                # plt.show()
                featureMap,points = feature_points_finding(self.segmentor,image, self.thresh,
                                                    zoneShape=self.outputZoneShape, n_clusters=self.n_clusters, 
                                                    normalize_GT=self.normalize_GT,idx=str(self.imagesList[idx]),mask_in = mask/255, gaussian_contour=self.gaussian_contour)
            except:
                # print("nothing")
                return transforms.ToTensor()(image), torch.zeros((image.shape[0],image.shape[1],2),dtype=torch.float32)
        # featureMap=np.expand_dims(featureMap, 0)
        # plt.imshow(featureMap[:,:,0])
        # plt.imshow(featureMap[:,:,1],alpha=0.2)
        # plt.show()
        # image = self.res(image = image)['image']
        target = self.transforms(image = image, mask0= featureMap[:,:,0], mask1 = featureMap[:,:,1])
        if self.normalize:
            normalized_gaussian_mask = _normalize_tensor(target['mask0'])
            # plt.imshow(normalized_gaussian_mask.cpu().numpy())
            # plt.show()
            # print("manual gts", manual_gt_points.shape)
            # print("points", points.shape)
            return target['image']/255, {'teacher_output':torch.stack([normalized_gaussian_mask,target['mask1']],dim = 2).float(), 'manual_gt': manual_gt_points, 'teacher_points': points}
        return target['image']/255, torch.stack([target['mask0'],target['mask1']],dim = 2).float(), manual_gt_points, points



if __name__ == '__main__':
    # Usage examples
    # -- Example 1: evaluation loop for the dataset and the teacher model
    from utils.utils_new import evaluate_KeyPoints_dataset
    from models.SegModel import DeepLabV3
    from models.models_student import ResNet_VAE
    from matplotlib import pyplot as plt
    checkpoint = torch.load(r'checkpoint_epoch900_one_channel_gauss.pt')
    model_VAE = ResNet_VAE(model_name='resnet18')
    model_VAE.load_state_dict(checkpoint['model_state_dict'])  
    model_VAE.to('cuda') 
    model_seg = DeepLabV3(3).to('cuda')
    model_seg.load_state_dict(torch.load('fold0_model.pth'))
    model_seg.eval()
    # valData = VAEDataset_NoFlags(r'DATA_DIR,model_seg, model_seg_type='deeplab',outputZoneShape='DoubleChannel_AutoGaussian',thresh=10,augment=True,normalize=True, n_clusters=4)
    # r = evaluate_KeyPoints_dataset(
    #     model_seg,
    #     model_VAE,
    #     valData,
    #     3,
    #     'resnet18'
    # )

    # jsonObj = json.dumps({"folds":[r]},indent=4)
    # with open('sampple.json','a') as f:
    #     f.write(jsonObj)
    # with open('sampple.json','r+') as file:
    #     file_data = json.load(file)
    #     file_data['folds'].append(r)
    #     file.seek(0)
    #     json.dump(file_data,file,indent=4)
    # from utils_new import write_json
    # write_json(r,'results.json')

    # -- Example 2: visualize the generated labels from the teacher model and compare them to the manual GT points
    yolo_path = r'train22\weights\best.pt'
    model_yolo = YOLO(yolo_path)
    model_yolo.to('cuda')

    DATA_DIR= r'Dataset\Changing speed\40\output'
    # SEG_MODEL_PATH = 'fold0_model.pth'
    data = VAEDataset_NoFlags(DATA_DIR,model_yolo,augment=True, model_seg_type='yolo', gaussian_contour= True,
                      normalize=False,outputZoneShape='DoubleChannel_AutoGaussian',n_clusters=4)
    
    for i in range(100):
        img, mask, pts, gt_points = data[i+100]
        print(f'auao generate labels are as follows: {pts}')
        print(f'gt points are as follows: {gt_points}')
        # plt.show()
        print(img.shape)
        print(mask.shape)
        
        plt.figure().clear()
        plt.close()
        plt.cla()
        plt.clf()
        # plt.imshow(mask[:,:,1].cpu().numpy(),alpha=0.5)
        plt.imshow(mask[:,:,0].cpu().numpy(),alpha=0.5)
        plt.imshow(img.permute(1,2,0).cpu().numpy(),alpha=0.5)
        heatmap = mask[:,:,0].cpu().numpy()
        binary = heatmap > 0.3

        labeled = measure.label(binary)
        regions = measure.regionprops(labeled,intensity_image=heatmap)
        """going beyond this ne sert à rien"""
        centers = [center_of_mass(region.intensity_image) for region in regions]
        print("Centers of mass:",centers)
        # plt.show()


        for props in regions:
            print(props.centroid)
            X, Y = props.centroid
            plt.scatter(Y,X, c = 'black')
        # for X,Y in centers:
        #     print(X,Y)
        #     plt.scatter(X,Y)

        plt.show()

    #     # plt.imshow(binary)
    #     # plt.show()
    #     print(torch.Tensor(img).dtype)
    #     print(torch.Tensor(mask).dtype)
    #     print()
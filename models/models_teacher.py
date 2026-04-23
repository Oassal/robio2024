import torch
import torchvision
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.autograd import Variable
from utils.utils_new import feature_points_finding
from utils.yolo_functions import extract_masks_with_tracking
from models.SegModel import DeepLabV3
import numpy as np
import cv2
from matplotlib import pyplot as plt

class Reshape(nn.Module):
    def __init__(self, *args) -> None:
        super().__init__()
        self.shape=args
    def forward(self,x):
        return x.view(self.shape)

class Trim(nn.Module):
    def __init__(self, *args):
        super().__init__()

    def forward(self, x):
        return x[:, :, :270, :360]
    
class Teacher(nn.Module):
    # TODO also add the possibility of using foundation models, such as DINOv2 + SAM
    def __init__(self, segmentation_model,
                 seg_model_name = "deeplabv3_resnet50",
                 heatmaps_finder = feature_points_finding,
                 nbPts = 4,
                 #segmentation_model,
                 heatmap_threshold = 1,
                 nb_clusters = None,
                 normalize_GT= True,
                 gaussian_contour = False,
                 outputZone_shape = "DoubleChannel_AutoGaussian",
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        """
        Teacher model for knowledge distillation.
        It uses a pre-trained DeepLabV3 model as a feature extractor and a heatmaps generation algorithm.
        """
        self.segmentation_model = segmentation_model
        self.nbPts = nbPts
        self.heatmap_threshold = heatmap_threshold
        self.nb_clusters = nb_clusters
        self.normalize_GT = normalize_GT
        # The output of the heatmap finder should always be a heatmap in 2D
        self.heatmaps_finder = heatmaps_finder
        self.segmentation_model.eval()  # Freeze the segmentation model
        self.seg_model_name = seg_model_name
        self.gaussian_contour = gaussian_contour
        self.idx = 0
        self.outputZone_shape = outputZone_shape


    def forward(self, x:torch.tensor):
        # future amelioration : add the possibility to use the teacher model for inference only, in which case we can use a random number of clusters for each image, and not have it as an argument of the class

        # Check if the input is a batch of images or a single image, and add a batch dimension if necessary
        if x.dim() == 3:
            x = x.unsqueeze(0)

        # Check if random number of clusters or not
        self.idx += 1
        if self.nb_clusters is None:
            self.nb_clusters = torch.randint(2, 6, (1,)).item()
        heatmaps = []  # Initialize heatmaps as an empty array
        with torch.no_grad():
            # Condition this later depending on the segmentation model that is used
            if self.seg_model_name == "deeplab":
                mask = self.segmentation_model(x)['out']
                mask_pool = torch.softmax(mask, dim=1)
                mask_pool = mask_pool[:,1,:,:]
                mask_arg = torch.argmax(mask, dim=1).unsqueeze(1).float()
                # future amelioration repalace arguments by kwargs
                batch_size = x.shape[0]
                for i in range(batch_size):
                    img = x[i,:,:,:]
                    mask_pool_i = mask_pool[i,:,:]
                    heatmap, points = self.heatmaps_finder(self.segmentation_model,
                                                            torch.permute(img,(2,1,0)).cpu().numpy(),
                                                            self.heatmap_threshold,
                                                            zoneShape=self.outputZone_shape,
                                                            n_clusters=self.nb_clusters,
                                                        normalize_GT=self.normalize_GT,
                                                        mask_in=mask_pool_i.cpu().numpy()>0.1,
                                                        idx=0
                                                        )
                    heatmaps.append(heatmap)
            # segm = F.interpolate(segm, size=(270, 360), mode='bilinear', align_corners=False)
                
            elif self.seg_model_name == "yolo":
                # future amelioration repalace arguments by kwargs
                masks = extract_masks_with_tracking(x.cpu().numpy(), self.idx, model=self.segmentation_model)
                try:
                    mask = masks['wire']
                    heatmap, points = self.heatmaps_finder(self.segmentation_model,
                                                            x.cpu().numpy(),
                                                            self.heatmap_threshold,
                                                            self.outputZone_shape,
                                                            self.nb_clusters,
                                                            self.normalize_GT,
                                                            idx = self.idx,
                                                            mask_in = mask/255,
                                                            gaussian_contour = self.gaussian_contour)
                                                            # nbPts=self.nbPts,
                                                            # threshold=self.heatmap_threshold,
                                                            # nb_clusters=self.nb_clusters,
                                                            # normalize=self.normalize_GT)
                    heatmaps.append(heatmap)
                except KeyError:
                    heatmap, points = None, None
            # Do the same thing for YOLO
            
            # From the mask, extract the heatmaps
            # heatmaps = self.heatmaps_finder(mask,
            #                                nbPts=self.nbPts,
            #                                threshold=self.GT_threshold,
            #                                nb_clusters=self.nb_clusters,
            #                                normalize=self.normalize_GT)
            
        heatmaps = np.array(heatmaps)
        # Return the heatmaps with the mask as well for training purposes
        heatmaps = torch.from_numpy(heatmaps).to(x.get_device())
        # return torch.cat([heatmaps, mask], dim=1)
        heatmaps = torch.permute(heatmaps, (0, 3, 1, 2))
        torch.permute(heatmaps, (0, 3, 1, 2))
        return heatmaps




if __name__ == "__main__":
    seg_model = DeepLabV3(3).to('cuda')
    seg_model.load_state_dict(torch.load("fold0_model.pth"))
    seg_model.eval()


    teacher_model = Teacher(segmentation_model = seg_model,
                            seg_model_name="deeplab",
                            heatmaps_finder=feature_points_finding,
                            )
    teacher_model = teacher_model.to('cuda')
    img = torch.rand((1,3,270,360)).to('cuda')
    img = cv2.imread("gens_big_dataset\img_init.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (360,270))
    # img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to('cuda')
    img = torch.from_numpy(img).permute(2, 0, 1).to('cuda')
    img = img.float()/255.
    # img = F.interpolate(img, size=(270, 360), mode='bilinear', align_corners=False)
    imgs = torch.stack([img,img,img],dim=0)
    print(imgs.shape)
    heatmaps = teacher_model(img)
    print(heatmaps.shape)
    import matplotlib.pyplot as plt
    # plt.imshow(heatmaps[0,:,:,0].cpu().numpy())
    plt.imshow(heatmaps[0,0,:,:].cpu().numpy())
    plt.show()

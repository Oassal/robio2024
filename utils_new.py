import torch
import torch.utils
import torch.utils.data
from utils.normals import find_extreme_point, find_cetnroid
from models.SegModel import DeepLabV3
import albumentations as alb
from torchvision import transforms
from matplotlib import pyplot as plt
import cv2
import numpy as np
import collections
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
from time import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models.models_student import VAE, ResNet_VAE
from scipy.spatial.distance import cdist
import random
import json
from pathlib import Path
import os

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

class ToTensor(alb.ImageOnlyTransform):
    def __init__(self):
        super().__init__(always_apply=True)

    def apply(self, image, **params):
        return transforms.ToTensor()(image)

    def get_params(self):
        return {}


transf = alb.Compose([
    alb.Resize(1080//4,1440//4, always_apply=True),
    ToTensor(),
])

transf2 = alb.Compose([
    alb.Resize(1080//4,1440//4, always_apply=True),
    ToTensor(),
])

def create_list_directories(mainDir: str):
    dirList=[]
    mainDir=Path(mainDir)
    for dir in sorted(mainDir.glob("**/")):
        # print(dir)
        if 'output' in list(dir.relative_to(mainDir).parts):
            if len(dir.relative_to(mainDir).parts) >0 :
                index_keyPtsData = dir.relative_to(mainDir).parts.index('output')
                # print(index_keyPtsData)
                dirKeyPoints = mainDir.joinpath(*dir.relative_to(mainDir).parts[:index_keyPtsData])/Path("data")
                # print(list(dir.relative_to(mainDir).parts))
                dirList.append({'out':str(dir), 'data':str(dirKeyPoints)})
                print(dir)
                # print(mainDir/dir)

    return dirList
def inference_VAE(model, device, img:np.array,)->np.array:
    '''
    takes as input a VAE model and an image and outputs the predicted map of feature points
    '''
    model.eval()
    with torch.no_grad():
        img_gpu  = transf2(image = img).pop('image').unsqueeze(0).to(device)
        start = time()
        out,_,_,out2 = model(img_gpu)
        end = time()
        # print(f'VAE inference time : {end-start}')
        features_predicted = torch.squeeze(out,dim=0).permute(1,2,0).cpu().numpy()
    return features_predicted

def _calculate_covariance(pointsDict:dict)->list:
    '''
    takes as input a set of points as dict, for each element in the dict finds the covariance matrix
    '''
    covDict = collections.defaultdict(list)
    for i in range(len(pointsDict)):
        pointsList = pointsDict[str(i)]
        # print(pointsList)
        # print(np.cov(np.array(pointsList).T))
        if len(pointsList) == 1 : 
            covDict[str(i)].append(([[20, 25],[20, 25]], len(pointsList)))
            continue
        covDict[str(i)].append((np.cov(np.array(pointsList).T), len(pointsList)))
        # print(covDict)
    
    return covDict 


def _gaussian_feature(Points:list,CovMatrix,ImgSize:tuple,normalize = True)->np.ndarray:
    """
    This function calculates the gaussian distribution for a set of points in an image
    Inputs : list of the cluster centers, covariance matrices for each cluster, and the size
    of image.
    Output : global heatmap describing all the clusters
    """
    x,y = np.mgrid[0:ImgSize[0]:1,0:ImgSize[1]:1]
    pos=np.dstack((x,y))

    gaussianDistribution = np.zeros((ImgSize[0],ImgSize[1]),dtype=np.float32)
    for i, (Xc,Yc) in enumerate(Points):
        try:
            rv = multivariate_normal([Yc,Xc],CovMatrix[str(i)][0][0],allow_singular=False)
        except:
            # print("except")
            rv = multivariate_normal([Yc,Xc],[[20, 25],[20, 25]])

        #normalize between 0 and 1
        if normalize:
            pdf = rv.pdf(pos)
            pdf -= np.min(pdf)
            pdf /=np.max(pdf)

            gaussianDistribution +=pdf
        else:
            gaussianDistribution+=rv.pdf(pos)*1 #CovMatrix[str(i)][0][1]
        #CovMatrix[str(i)][0][1]
        # print('img done')
    return gaussianDistribution

def feature_points_finding(model:DeepLabV3, img, thresh, zoneShape = 'Gaussian', n_clusters=4, normalize_GT = True, idx = None, mask_in = None, gaussian_contour = True)->tuple:
    # start = time()
    kmeans = KMeans(n_clusters=n_clusters, n_init=10,init='k-means++')
    labeledPointsDict=collections.defaultdict(list)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # print('img start')
    # model.to(device)
    # model.eval()
    # kstart = time()
    if mask_in is None:
        img_gpu  = transf(image = img).pop('image').unsqueeze(0).to(device)
        
        # plt.imshow(torch.permute(img_gpu.squeeze(0),dims=(1,2,0)).cpu().numpy())
        # plt.show()
        # print(img_gpu.shape)
        with torch.no_grad():
            out_segmentation = model(img_gpu)['out'].squeeze(0)
        all_masks = torch.softmax(out_segmentation,dim = 0).cpu().numpy()>0.1
        tmp_img = all_masks[1]
    else :
        tmp_img = mask_in
    # plt.imshow(tmp_img)
    # plt.show()
    # kend = time()
    # plt.imshow(tmp_img*255)
    # plt.show()
    # plt.imsave('temp/tmp_mask.png',tmp_img*255)
    # plt.imsave('temp/tm_image_raw.png',img_gpu.cpu().squeeze(0).permute(1,2,0).numpy())

    width, height = tmp_img.shape
    # print(width,height)
    mask = np.zeros(shape =(width,height,3))
    mask = mask.astype(np.uint8)
    mask[:,:,2]=tmp_img*255
    mask[:,:,0]=tmp_img*255
    mask[:,:,1] = tmp_img*255
    mask.astype(np.uint8)
    # plt.imshow(mask,alpha=0.5)
    # plt.show()
    # print(mask.shape)
    # tmp_mask = cv2.imread('temp/tmp_mask.png')
    # tmp_mask = mask
    # mask = mask.astype(np.uint8)
    points = find_extreme_point(mask,thresh=thresh)
    # Generate a probabilistic representation of the points
    
    if gaussian_contour:
        prob_points = []
        for point in points:
            x, y = point
            # Add some random noise to create a probabilistic representation
            x_prob = np.random.normal(loc=x, scale=5)  # Mean at x, standard deviation of 5
            y_prob = np.random.normal(loc=y, scale=5)  # Mean at y, standard deviation of 5
            prob_points.append([x_prob, y_prob])
        # kstart = time()
        points = prob_points
    colors = ['b', 'g', 'r','c','m','y']
    if len(points)==1:
        for i in range(len(points)):
            # plt.imsave('tmp/error.png',img)
            plt.scatter(points[i][0], points[i][1], color='g')
        points = [[0,0],[100,100],[200,200],[300,300]]
        print(idx)
        # plt.show()
    kmeans.fit(points)
    # kend = time()
    # print(f'fitting time {kend-kstart}')
    colors = ['b', 'g', 'r','c','m','y']
    for i in range(len(points)):
        # plt.scatter(points[i][0], points[i][1], color=colors[kmeans.labels_[i]])
        labeledPointsDict[str(kmeans.labels_[i])].append([points[i][1], points[i][0]])
    # print(labeledPointsDict)
    # plt.show()
    #covariance matrix per cluster, this is a dict
    covarianceMatrix = _calculate_covariance(labeledPointsDict)
    # show the plot
    # plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],color = 'y',s=100)
    # plt.xlim(0,img.shape[1]//4)
    # plt.ylim(0,img.shape[0]//4)
    # plt.gca().invert_yaxis()
    # plt.show() 
    # print(kmeans.labels_)

    '''add predict on points variable, get output as dictionary maybe to be used later on for cov
    calculation. add the points in a dictionary that is labele by cluster number maybe, and maybe
    create a function that returns this dictionary for better structuring of the code
    add calculatedCov variable that calculated covariance of each cluster using np
    '''



    # two_channels_output = np.stack([100*_gaussian_feature(points, covarianceMatrix, (width,height)), tmp_img], axis=2)
    # two_channels_output = np.stack([100*_gaussian_feature(points, [[20, 25],[2dd0, 25]], (width,height)), tmp_img,np.zeros_like(tmp_img)], axis=2)
    # end = time()
    # print(f'elapsed time : {end-start} ----- elapsed time for kMean : {kend-kstart}')
    if zoneShape == 'SingleChannel_Gaussian':
        one_channel_output_auto = 1*_gaussian_feature(kmeans.cluster_centers_,
                                    covarianceMatrix,(width,height))
        # return _gaussian_feature(points, [[20, 25],[20, 25]], (width,height)), points
        return one_channel_output_auto, points
    
    # elif zoneShape == 'DoubleChannel_Gaussian':
    #     return two_channels_output, points
    
    elif zoneShape == '3Channels-_Circular':
        for Xc, Yc in points:
            cv2.circle(mask,(Xc,Yc),1,(255,255,0),3)
    elif zoneShape == 'DoubleChannel_AutoGaussian':

        ######### np.append(a,[[find_cetnroid(mask)[0],find_cetnroid(mask)[1]]],axis = 0) right syntax to insert the centroid into the list of points. To do, append the value of the covariance
        ######### for a single pooint (centroid to the list and retarain)
        ######### alternative solution is to sum up the gaussian distribution directly on the cntroid with the distribution previously found frm the function (previously rv)
        ######### so at the end it will be somethign like : rv = _gaussian_feature(kmeans,covMat,(wid,height),img) + guass dist for single point at the center with teh 25 25 cov matrix
        '''todo'''
        two_channels_output_auto = np.stack([_gaussian_feature(kmeans.cluster_centers_,
                                    covarianceMatrix,(width,height),normalize = normalize_GT),tmp_img], axis = 2)
        return two_channels_output_auto, kmeans.cluster_centers_
        return mask, points
    else:
        raise ValueError

def metric(mask, features, plot = True):
    '''a function that takes as input the mask of an image and the predicted heatmap,
    finds the average distance between each of the points and the contour ??? the smaller the better ??'''
    gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    contours,_ = cv2.findContours(gray_mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    try:
        contour = max(contours, key = cv2.contourArea)
    except:
        return 0
    average_distance = 0
    
    x_featuers, y_features = np.nonzero(features)

    for x,y in zip(x_featuers,y_features):
        if not contours:
            average_distance+=0
        else:
            average_distance+=cv2.pointPolygonTest(contour,(int(x),int(y)),True)
    if len(x_featuers) == 0:
        return 0
    average_distance/= len(x_featuers)
    return average_distance

def metric_clustersCenters(image, segModel,thresh, zoneShape, features, featuresThresh, plot = True):
    '''
    Takes as input an image, and the corresponding features found from the ResNet-VAE and
    calculates the distance between the clutser wchich is considered to be the metric
    to do : make it output the average, and the values for each point; in a dictionary form
    and add inference time maybe for each operation ?? 
    '''
    start = time()
    _,clustersCenters = feature_points_finding(segModel,image,thresh,zoneShape)
    end = time()
    # print(f'Segmentation + Kmeans : {end-start}')
    ###clustering the points in the features prediction image
    features_thresholded = features[:,:,0]>featuresThresh
    x=[]
    y=[]
    points=[]
    for i in range(features_thresholded.shape[0]):
        for j in range(features_thresholded.shape[1]):
            if features_thresholded[i][j]:
                points.append([i,j])
                x.append(j)
                y.append(i)
            else:
                continue
    # plt.scatter(x,y,s=1)
    # plt.xlim(0,features_thresholded.shape[1])
    # plt.ylim(0,features_thresholded.shape[0])
    # plt.gca().invert_yaxis()
    # plt.show()
    '''measure time starting from here and sum it up with VAE inference'''
    X= points
    t1= time()
    kmeans = KMeans(n_clusters=4,n_init='auto')
    kmeans.fit(X)
    t2= time()
    # colors = ['b','g','r','c','m','y']
    # for i in range(len(X)):
    #     plt.scatter(X[i][1],X[i][0], color = colors[kmeans.labels_[i]], s =1)

    # plt.scatter(kmeans.cluster_centers_[:,1],kmeans.cluster_centers_[:,0],color = 'y')
    # # plt.scatter(clustersCenters[:,0],clustersCenters[:,1], color = 'b')
    # plt.xlim(0,features_thresholded.shape[1])
    # plt.ylim(0,features_thresholded.shape[0])
    # plt.gca().invert_yaxis()
    # plt.show()
    # print(clustersCenters)
    # print(np.flip(kmeans.cluster_centers_))
    A = clustersCenters
    B = np.flip(kmeans.cluster_centers_)
    if plot:
        plt.subplot(3,1,3)
        for i in range(len(kmeans.cluster_centers_)):
            plt.scatter(clustersCenters[i][0],clustersCenters[i][1], color = 'red', s=4)
        plt.subplot(3,1,2)
        for i in range(len(kmeans.cluster_centers_)):
            plt.scatter(clustersCenters[i][0],clustersCenters[i][1], color = 'red', s=4)
    dist_matrix = cdist(A,B,'euclidean')
    # print(dist_matrix)
    min_distance = np.min(dist_matrix,axis=1)
    # print(min_distance)
    # print(np.mean(min_distance))
    # print(t2-t1)
    return {
        'avgDistance' : np.mean(min_distance),
        'avgDistance_perPoint': min_distance
    }

def metric_GT(image, clustersGT,model,device = 'cuda',feature_thresh=0.3, plot = False,nb_clusters = 4):

    img_features = inference_VAE(model,device,image)
    features_thresholded = img_features[:,:,0]>feature_thresh
    if plot: 
        plt.subplot(3,1,2)
        plt.axis("off")
        plt.imshow(img_features[:,:,0], alpha=1)

        plt.subplot(3,1,3)
        plt.axis("off")
        plt.imshow(img_features[:,:,0], alpha = 0.5)
    x=[]
    y=[]
    points = []
    for i in range(features_thresholded.shape[0]):
        for j in range(features_thresholded.shape[1]):
            if features_thresholded[i][j]:
                points.append([i,j])
                x.append(i)
                y.append(j)
            else:
                continue
    if not points:
        return 0
    X = points
    kmeans = KMeans(n_clusters=nb_clusters,n_init='auto')
    kmeans.fit(X)
    A = clustersGT
    B = np.flip(kmeans.cluster_centers_)
    # print(f'A is : {A}, B is {B}')
    dist_matrix=cdist(A,B)
    min_distance = np.min(dist_matrix,axis=1)   
    if plot: 
        plt.subplot(3,1,2)
        plt.axis("off")
        for i in range(len(kmeans.cluster_centers_)):
            plt.scatter(kmeans.cluster_centers_[i][1],kmeans.cluster_centers_[i][0], color = 'r', s=5)
        plt.subplot(3,1,3)
        for i in range(len(kmeans.cluster_centers_)):
            print(kmeans.cluster_centers_[i][0])
            print(kmeans.cluster_centers_[i][1])
            plt.scatter(kmeans.cluster_centers_[i][1],kmeans.cluster_centers_[i][0], color = 'black', s=4)
        plt.text(2,250,f'mED = {np.mean(min_distance):.2f}', color = 'w')
        plt.subplot(3,1,2)
        for i in range(len(kmeans.cluster_centers_)):
            print(kmeans.cluster_centers_[i][0])
            print(kmeans.cluster_centers_[i][1])
            plt.scatter(kmeans.cluster_centers_[i][1],kmeans.cluster_centers_[i][0], color = 'black', s=4)
    # dist_matrix=cdist(A,B)
    # min_distance = np.min(dist_matrix,axis=1)
    # print(min_distance)
    # print(np.mean(min_distance))
    # for i,point in enumerate(kmeans.cluster_centers_):
    #     plt.text(point[1],point[0],f'ED{i}', color = 'w', fontsize = 10)
        s = f''
        for i,distance in enumerate(min_distance):
            s+=f'ED{i+1} : {distance:.1f} \n'
        plt.text(2,135,s,color = 'w', fontsize = 7)
    
    return {
        'avgDistance': np.mean(min_distance),
        'avgDistance_perPoint': min_distance
    }
        

def make_video(dataDir,saveDir,modelVAE, modelSeg):

    '''
    a function that makes a video from a datset of images,
    copy the code from below that does plot and sublopt,
    and also maybe add to it error (mean), and GT from auto labeling
    '''


def evaluate_KeyPoints(model_seg,
                       path_vae,
                       dir_data,
                       architecture_VAE = 'resnet50',
                       video_Write = False
                       ):
    """
    Inputs : model_seg : the used segmentation moedl, 
    path_vae : the path to the VAE network weights,
    dir_data: the path to the data, 
    architecture_VAE : the used architecture of VAE, video_Write: whether a video is needed or not.
    """

    """
    modify this to use a dataset instead of a directory ??
    Dataset would allow for data augmentation, in which we could add shear and stuff
    in this case we will need another dataset class that reads only keypoints, and not the
    original one
    maybe seperate both functions : comparison with automatic label and with manual labels


    use kwargs, pass either the path or dataset. If path is passed, ez. If dataset is passed
    do otherwise.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_VAE =ResNet_VAE(model_name=architecture_VAE)
    pdir = Path(dir_data)
    #VAE checkpoint
    checkpoint = torch.load(path_vae)
    model_VAE.load_state_dict(checkpoint['model_state_dict'])
    model_VAE.to(device)


    ##Initialize variables

    j=0
    #Fail rate of manual labels
    fails_15 = 0
    fails_10 = 0
    fails_5 = 0
    fails_20=0
    
    #Fail rate of automatic labels
    fails_5_auto=0
    fails_10_auto=0
    fails_15_auto=0
    fails_20_auto=0

    #initialize arrays used for error calculation
    Y_plot = []
    error_p1_ManualGT=[]
    error_p2_ManualGT=[]
    error_p3_ManualGT=[]
    error_p4_ManualGT=[]
    avgError_ManualGT = []

    error_p1_AutoGT=[]
    error_p2_AutoGT=[]
    error_p3_AutoGT=[]
    error_p4_AutoGT=[]
    avgError_AutoGT=[]

    #resize factor in case of video writing
    if video_Write:
        resizeFactor = 4
        out = cv2.VideoWriter(f'Video_{architecture_VAE}.avi', cv2.VideoWriter_fourcc('M','J','P','G'),
                              15, (resizeFactor*164,resizeFactor*389))
        
    for jsonFile in sorted(pdir.rglob("*.json")):
        j+=1
        centers= []
        mean = 0
        with open(jsonFile,"r") as f:
            data = json.load(f)
            if not data['tooltips']:
                continue
            else:
                for i,key in enumerate(data['tooltips']):
                    centers.append([key['x'], key['y']])
            if len(centers)!=4:continue

            imgPath = str(jsonFile).replace(".json",".jpg")
            img = cv2.imread(imgPath)

            if img.shape [0]> 500:
                continue
            
            #check if we want to generate a video or we want just to evaluate
            if video_Write:
                plt.subplot(3,1,1)
                plt.axis("off")
                plt.imshow(img)
                plt.subplot(3,1,2)
                plt.subplot(3,1,3)
                plt.imshow(img)
                for i in range(len(centers)):
                    plt.scatter(centers[i][0],centers[i][1], color = 'orange', s=4)
                plt.subplot(3,1,2)
                for i in range(len(centers)):
                    plt.scatter(centers[i][0],centers[i][1], color = 'orange', s=4)  


            result = metric_GT(img,centers,model_VAE,device, plot=video_Write)
            a = result['avgDistance']
            if a>5:
                fails_5 +=1
            if a>10:
                fails_10 +=1
            if a > 15:
                # print(a)
                fails_15+=1
                # print(f'fails : {fails}/{leng}')
            if a>20:
                fails_20+=1
            

            error_p1_ManualGT.append(result['avgDistance_perPoint'][0])
            error_p2_ManualGT.append(result['avgDistance_perPoint'][1])
            error_p3_ManualGT.append(result['avgDistance_perPoint'][2])
            error_p4_ManualGT.append(result['avgDistance_perPoint'][3])
            avgError_ManualGT.append(result['avgDistance'])

            if video_Write:
                plt.savefig(f"tmp/3imgs.png",bbox_inches='tight')
                im = cv2.imread('tmp/3imgs.png')
                im = cv2.resize(im,(resizeFactor*164,resizeFactor*389))
                out.write(im)
                # plt.show()
                plt.figure().clear()
                plt.cla()
                plt.clf()
            
            features= inference_VAE(model_VAE,device,img)
            d = metric_clustersCenters(img,model_seg,2,'DoubleChannel_AutoGaussian',
                                        features, 0.1,plot=video_Write)
            error_p1_AutoGT.append(d['avgDistance_perPoint'][0])
            error_p2_AutoGT.append(d['avgDistance_perPoint'][1])
            error_p3_AutoGT.append(d['avgDistance_perPoint'][2])
            error_p4_AutoGT.append(d['avgDistance_perPoint'][3])
            avgError_AutoGT.append(d['avgDistance'])
            if d['avgDistance']>5:
                fails_5_auto+=1
            if d['avgDistance']>10:
                fails_10_auto+=1
            if d['avgDistance']>15:
                fails_15_auto+=1
            if d['avgDistance']>20:
                fails_20_auto+=1



    avgError_p1_autoGT= np.mean(error_p1_AutoGT)
    avgError_p2_autoGT= np.mean(error_p2_AutoGT)
    avgError_p3_autoGT= np.mean(error_p3_AutoGT)
    avgError_p4_autoGT= np.mean(error_p4_AutoGT)
    avgDatasetError_autoGT = np.mean(avgError_AutoGT)
    standardDeviation_autoGT = np.std(avgError_AutoGT)

    avgError_p1_ManualGT= np.mean(error_p1_ManualGT)
    avgError_p2_ManualGT= np.mean(error_p2_ManualGT)
    avgError_p3_ManualGT= np.mean(error_p3_ManualGT)
    avgError_p4_ManualGT= np.mean(error_p4_ManualGT)
    avgDatasetError_ManualGT = np.mean(avgError_ManualGT)
    standardDeviation_ManualGT = np.std(avgError_ManualGT)

    print(f'for autoGT the average error is {avgDatasetError_autoGT} \
          and per point errors are : avgError P1 : {avgError_p1_autoGT}, \
          avgError P2 : {avgError_p2_autoGT},  \
          avgError P3 : {avgError_p3_autoGT},  \
          avgError P4 : {avgError_p4_autoGT}, \
          std Dev: {standardDeviation_autoGT} ')

    print(f'for ManualGT the average error is {avgDatasetError_ManualGT} \
          and per point errors are : avgError P1 : {avgError_p1_ManualGT},\
          avgError P2 : {avgError_p2_ManualGT}, \
          avgError P3 : {avgError_p3_ManualGT}, \
          avgError P4 : {avgError_p4_ManualGT}, \
          std Dev: {standardDeviation_ManualGT} ')
    
    print('----------------Manually generated labels----------------')
    print(f'mean euclidean error : {mean/j}')
    print(f'under 5 : {fails_5}/{j}, rate: {fails_5/j} ------')
    print(f'under 10 : {fails_10}/{j}, rate: {fails_10/j} ------')
    print(f'under 15 : {fails_15}/{j}, rate: {fails_15/j} ------')
    print(f'under 20 : {fails_20}/{j}, rate: {fails_20/j} ------')

    print('----------------Automatically generated labels----------------')
    print(f'under 5 : {fails_5_auto}/{j}, rate: {fails_5_auto/j} ------')
    print(f'under 10 : {fails_10_auto}/{j}, rate: {fails_10_auto/j} ------')
    print(f'under 15 : {fails_15_auto}/{j}, rate: {fails_15_auto/j} ------')
    print(f'under 20 : {fails_20_auto}/{j}, rate: {fails_20_auto/j} ------')

    print('----------------Done----------------')

    result = {
'autoGT' : {    
            'autoGT_mED': avgDatasetError_autoGT,
            'mED_P1': avgError_p1_autoGT,
            'mED_P2': avgError_p2_autoGT,
            'mED_P3': avgError_p3_autoGT,
            'mED_P4': avgError_p4_autoGT,
            'stdDev': standardDeviation_autoGT
                },
'manualGT':{
            'manualGT_mED':avgDatasetError_ManualGT,
            'mED_P1':avgError_p1_ManualGT,
            'mED_P1':avgError_p2_ManualGT,
            'mED_P1':avgError_p3_ManualGT,
            'mED_P1':avgError_p4_ManualGT,
            'stdDev': standardDeviation_ManualGT
                },
'PmED_autoGT':{
    'PmED5': 1 - fails_5_auto/j,
    'PmED10': 1 - fails_10_auto/j,
    'PmED15': 1 - fails_15_auto/j,
    'PmED20': 1 - fails_20_auto/j
},
'PmED_manualGT': {
    'PmED5': 1 - fails_5/j,
    'PmED10': 1 - fails_10/j,
    'PmED15': 1 - fails_15/j,
    'PmED20': 1 - fails_20/j
}
    }

    return result



'''evaluate using dataset and not directory'''
def evaluate_KeyPoints_dataset(model_seg,
                       model_vae,
                       val_data:torch.utils.data.Dataset,
                       epoch,
                       architecture_VAE = 'resnet50_AE',
                       video_Write = False
                       ):
    """
    Inputs : model_seg : the used segmentation moedl, 
    path_vae : the path to the VAE network weights,
    dir_data: the path to the data, 
    architecture_VAE : the used architecture of VAE, video_Write: whether a video is needed or not.
    """

    """
    modify this to use a dataset instead of a directory ??
    Dataset would allow for data augmentation, in which we could add shear and stuff
    in this case we will need another dataset class that reads only keypoints, and not the
    original one
    maybe seperate both functions : comparison with automatic label and with manual labels


    use kwargs, pass either the path or dataset. If path is passed, ez. If dataset is passed
    do otherwise.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ##Initialize variables
    j=0
    #Fail rate of manual labels
    fails_15 = 0
    fails_10 = 0
    fails_5 = 0
    fails_20=0
    
    #Fail rate of automatic labels
    fails_5_auto=0
    fails_10_auto=0
    fails_15_auto=0
    fails_20_auto=0

    #initialize arrays used for error calculation
    Y_plot = []
    error_p1_ManualGT=[]
    error_p2_ManualGT=[]
    error_p3_ManualGT=[]
    error_p4_ManualGT=[]
    avgError_ManualGT = []

    error_p1_AutoGT=[]
    error_p2_AutoGT=[]
    error_p3_AutoGT=[]
    error_p4_AutoGT=[]
    avgError_AutoGT=[]

    #resize factor in case of video writing
    if video_Write:
        resizeFactor = 4
        out = cv2.VideoWriter(f'Video_{architecture_VAE}_{epoch}.avi', cv2.VideoWriter_fourcc('M','J','P','G'),
                              15, (resizeFactor*164,resizeFactor*389))
        
    for img,target_points in val_data:
        j+=1
        if j<70: continue
        else: nb_clusters=7
        # if j == 5:break
        # print(target_points)
        centers= target_points
        mean = 0  
        img = img.permute(1,2,0).cpu().numpy()
        #check if we want to generate a video or we want just to evaluate
        if video_Write:
            plt.subplot(3,1,1)
            plt.axis("off")
            plt.imshow(img)
            plt.subplot(3,1,2)
            plt.subplot(3,1,3)
            plt.imshow(img)
            for i in range(len(centers)):
                plt.scatter(centers[i][0],centers[i][1], color = 'orange', s=4)
            plt.subplot(3,1,2)
            for i in range(len(centers)):
                plt.scatter(centers[i][0],centers[i][1], color = 'orange', s=4)  


        result = metric_GT(img,target_points,model_vae,device,feature_thresh=0.05, plot=video_Write,nb_clusters=nb_clusters)
        print(result)
        a = result['avgDistance']
        if a>5:
            fails_5 +=1
        if a>10:
            fails_10 +=1
        if a > 15:
            # print(a)
            fails_15+=1
            # print(f'fails : {fails}/{leng}')
        if a>20:
            fails_20+=1
        

        error_p1_ManualGT.append(result['avgDistance_perPoint'][0])
        error_p2_ManualGT.append(result['avgDistance_perPoint'][1])
        error_p3_ManualGT.append(result['avgDistance_perPoint'][2])
        error_p4_ManualGT.append(result['avgDistance_perPoint'][3])
        avgError_ManualGT.append(result['avgDistance'])

        if video_Write:
            plt.savefig(f"tmp/3imgs.png",bbox_inches='tight')
            plt.savefig(f"video_dataset/frame_{j}.eps",bbox_inches = 'tight')
            plt.savefig(f"video_dataset/frame_{j}.jpg",bbox_inches = 'tight')
            im = cv2.imread('tmp/3imgs.png')
            im = cv2.resize(im,(resizeFactor*164,resizeFactor*389))
            out.write(im)
            # plt.show()
            plt.figure().clear()
            plt.cla()
            plt.clf()
        
        features= inference_VAE(model_vae,device,img)
        d = metric_clustersCenters(img,model_seg,2,'DoubleChannel_AutoGaussian',
                                    features, 0.05,plot=video_Write)
        error_p1_AutoGT.append(d['avgDistance_perPoint'][0])
        error_p2_AutoGT.append(d['avgDistance_perPoint'][1])
        error_p3_AutoGT.append(d['avgDistance_perPoint'][2])
        error_p4_AutoGT.append(d['avgDistance_perPoint'][3])
        avgError_AutoGT.append(d['avgDistance'])
        if d['avgDistance']>5:
            fails_5_auto+=1
        if d['avgDistance']>10:
            fails_10_auto+=1
        if d['avgDistance']>15:
            fails_15_auto+=1
        if d['avgDistance']>20:
            fails_20_auto+=1



    avgError_p1_autoGT= np.mean(error_p1_AutoGT)
    avgError_p2_autoGT= np.mean(error_p2_AutoGT)
    avgError_p3_autoGT= np.mean(error_p3_AutoGT)
    avgError_p4_autoGT= np.mean(error_p4_AutoGT)
    avgDatasetError_autoGT = np.mean(avgError_AutoGT)
    standardDeviation_autoGT = np.std(avgError_AutoGT)
    standardDeviation_autoGT_P1 = np.std(error_p1_AutoGT)
    standardDeviation_autoGT_P2 = np.std(error_p2_AutoGT)
    standardDeviation_autoGT_P3 = np.std(error_p3_AutoGT)
    standardDeviation_autoGT_P4 = np.std(error_p4_AutoGT)

    avgError_p1_ManualGT= np.mean(error_p1_ManualGT)
    avgError_p2_ManualGT= np.mean(error_p2_ManualGT)
    avgError_p3_ManualGT= np.mean(error_p3_ManualGT)
    avgError_p4_ManualGT= np.mean(error_p4_ManualGT)
    avgDatasetError_ManualGT = np.mean(avgError_ManualGT)
    standardDeviation_ManualGT = np.std(avgError_ManualGT)
    standardDeviation_ManualGT_P1 = np.std(error_p1_ManualGT)
    standardDeviation_ManualGT_P2 = np.std(error_p2_ManualGT)
    standardDeviation_ManualGT_P3 = np.std(error_p3_ManualGT)
    standardDeviation_ManualGT_P4 = np.std(error_p4_ManualGT)

    print(f'for autoGT the average error is {avgDatasetError_autoGT} \
          and per point errors are : avgError P1 : {avgError_p1_autoGT}, \
          avgError P2 : {avgError_p2_autoGT},  \
          avgError P3 : {avgError_p3_autoGT},  \
          avgError P4 : {avgError_p4_autoGT}, \
          std Dev: {standardDeviation_autoGT} ')

    print(f'for ManualGT the average error is {avgDatasetError_ManualGT} \
          and per point errors are : avgError P1 : {avgError_p1_ManualGT},\
          avgError P2 : {avgError_p2_ManualGT}, \
          avgError P3 : {avgError_p3_ManualGT}, \
          avgError P4 : {avgError_p4_ManualGT}, \
          std Dev: {standardDeviation_ManualGT} ')
    
    print('----------------Manually generated labels----------------')
    print(f'mean euclidean error : {mean/j}')
    print(f'under 5 : {fails_5}/{j}, rate: {1-fails_5/j} ------')
    print(f'under 10 : {fails_10}/{j}, rate: {1-fails_10/j} ------')
    print(f'under 15 : {fails_15}/{j}, rate: {1-fails_15/j} ------')
    print(f'under 20 : {fails_20}/{j}, rate: {1-fails_20/j} ------')

    print('----------------Automatically generated labels----------------')
    print(f'under 5 : {fails_5_auto}/{j}, rate: {1-fails_5_auto/j} ------')
    print(f'under 10 : {fails_10_auto}/{j}, rate: {1-fails_10_auto/j} ------')
    print(f'under 15 : {fails_15_auto}/{j}, rate: {1-fails_15_auto/j} ------')
    print(f'under 20 : {fails_20_auto}/{j}, rate: {1-fails_20_auto/j} ------')

    print('----------------Done----------------')            
    result = {
        'autoGT' : {    
                    'autoGT_mED': avgDatasetError_autoGT,
                    'mED_P1': avgError_p1_autoGT,
                    'mED_P2': avgError_p2_autoGT,
                    'mED_P3': avgError_p3_autoGT,
                    'mED_P4': avgError_p4_autoGT,
                    'stdDev': standardDeviation_autoGT,
                    'stDev_P1':standardDeviation_autoGT_P1,
                    'stDev_P2':standardDeviation_autoGT_P2,
                    'stDev_P3':standardDeviation_autoGT_P3,
                    'stDev_P4':standardDeviation_autoGT_P4
                        },
        'manualGT':{
                    'manualGT_mED':avgDatasetError_ManualGT,
                    'mED_P1':avgError_p1_ManualGT,
                    'mED_P1':avgError_p2_ManualGT,
                    'mED_P1':avgError_p3_ManualGT,
                    'mED_P1':avgError_p4_ManualGT,
                    'stdDev': standardDeviation_ManualGT,
                    'stdDev_P1': standardDeviation_ManualGT_P1,
                    'stdDev_P2': standardDeviation_ManualGT_P2,
                    'stdDev_P3': standardDeviation_ManualGT_P3,
                    'stdDev_P4': standardDeviation_ManualGT_P4,
                    
                        },
        'PmED_autoGT':{
            'PmED5': 1 - fails_5_auto/j,
            'PmED10': 1 - fails_10_auto/j,
            'PmED15': 1 - fails_15_auto/j,
            'PmED20': 1 - fails_20_auto/j
        },
        'PmED_manualGT': {
            'PmED5': 1 - fails_5/j,
            'PmED10': 1 - fails_10/j,
            'PmED15': 1 - fails_15/j,
            'PmED20': 1 - fails_20/j
        }
    }

    return result

def save_results(
        results_dict,
        fileName,
):
    """
    This function is supposed to take as input the output of the validation function and to save them
    in a yaml/json file? a file of any type, most likely a yaml file.
    for each fold save the metric values and use another fucnction to plot them
    """

def write_json(
        data,
        path:str = None,
):
    if path == None: path = 'results.json'

    if not os.path.isfile(path):
        with open(path,'a') as file:
            jsonObj = json.dumps({"folds":[data]},indent=4)
            file.write(jsonObj)
    else:
        with open(path,'r+') as file:
            try:
                file_data = json.load(file)
                file_data['folds'].append(data)
                file.seek(0)
                json.dump(file_data,file,indent=4) 
            except json.JSONDecodeError:
                jsonObj = json.dumps({"folds":[data]},indent=4)
                file.write(jsonObj)

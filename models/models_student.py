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
    

class VAE(nn.Module):
    def __init__(self, nbOutputChannels = 1) -> None:
        '''
        Vanilla VAE model with a simple CNN encoder and decoder. The output of the decoder is a heatmap of the same size as the input image, with values between 0 and 1.
        '''
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3,32,stride=2, kernel_size=3,bias=False,padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Dropout2d(0.25),
            #
            nn.Conv2d(32,64,stride=2,kernel_size=3,bias=False,padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Dropout2d(0.25),
            #
            nn.Conv2d(64,64,stride=2,kernel_size=3,bias=False,padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Dropout2d(0.25),
            #
            nn.Conv2d(64,64,stride=2,kernel_size=3,bias=False,padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.25),
            #
            nn.Flatten(),
        )

        self.z_mean = torch.nn.Linear(25024,10)
        self.z_log_var= torch.nn.Linear(25024, 10)

        self.decoder = nn.Sequential(
            torch.nn.Linear(10,27648),
            Reshape(-1,64,18,24),
            #
            nn.ConvTranspose2d(64,64,stride=2,kernel_size=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Dropout2d(0.25),
            #
            nn.ConvTranspose2d(64,64,stride=2,kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Dropout2d(0.25),
            #
            nn.ConvTranspose2d(64,32,stride=2,kernel_size=3,padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Dropout2d(0.25),
            #
            nn.ConvTranspose2d(32,nbOutputChannels,stride=2,kernel_size=3,padding=1),
            #
            Trim(),
            nn.Sigmoid()
            # nn.Tanh()
        )

    def encoding_fn(self,x):
        x=self.encoder(x)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        encoded = self.reparametrize(z_mean,z_log_var)
        return encoded


    def reparameterize(self, z_mu, z_log_var):
        eps = torch.randn(z_mu.size(0), z_mu.size(1)).to(z_mu.get_device())
        # eps = torch.randn(z_mu.size(0), z_mu.size(1))
        z = z_mu + eps * torch.exp(z_log_var/2.)
        # z=z.to('cuda')
        return z


    def forward(self, x):
        x = self.encoder(x)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        encoded = self.reparameterize(z_mean, z_log_var)
        decoded = self.decoder(encoded)
        return encoded, z_mean, z_log_var, decoded


def get_pretrained_model(model_name):
    """Retrieve a pre-trained model from torchvision

    Params
    -------
        model_name (str): name of the model (currently only accepts vgg16 and resnet50)

    Return
    --------
        model (PyTorch model): cnn

    """

    if model_name == 'vgg16':
        # model = models.vgg16(pretrained=False)
        model = models.vgg16()
        # model.load_state_dict(torch.load('../input/vgg16/vgg16.pth'))

        # Freeze early layers
        for param in model.parameters():
            param.requires_grad = False
        #n_outputs = model.classifier[6].out_features

    elif model_name == 'resnet50':
        model = models.resnet50(torchvision.models.ResNet50_Weights)
        # model.load_state_dict(torch.load('../input/resnet50/resnet50.pth'))

        for param in model.parameters():
            param.requires_grad = True

        #n_outputs = model.fc.out_features
    elif model_name == 'resnet18':
        model = models.resnet18(torchvision.models.ResNet18_Weights)
        # model.load_state_dict(torch.load('../input/resnet50/resnet50.pth'))

        for param in model.parameters():
            param.requires_grad = True

        #n_outputs = model.fc.out_features
    elif model_name == 'resnet34':
        model = models.resnet34(torchvision.models.ResNet34_Weights)
        # model.load_state_dict(torch.load('../input/resnet50/resnet50.pth'))

        for param in model.parameters():
            param.requires_grad = True

        #n_outputs = model.fc.out_features

    return model#, n_outputs

class ResNet_VAE(nn.Module):
    def __init__(self, fc_hidden1=1024, fc_hidden2=768, drop_p=0.3, CNN_embed_dim=256,model_name='resnet50', NB_OUTPUT_CHANNELS=2):
        '''
        VAE model with a ResNet encoder and a simple CNN decoder. The output of the decoder is a heatmap of the same size as the input image, with values between 0 and 1.
        '''
        super(ResNet_VAE, self).__init__()

        self.fc_hidden1, self.fc_hidden2, self.CNN_embed_dim = fc_hidden1, fc_hidden2, CNN_embed_dim

        # CNN architechtures
        self.ch1, self.ch2, self.ch3, self.ch4 = 16, 32, 64, 128
        self.k1, self.k2, self.k3, self.k4 = (5, 5), (3, 3), (3, 3), (3, 3)      # 2d kernal size
        self.s1, self.s2, self.s3, self.s4 = (2, 2), (2, 2), (2, 2), (2, 2)      # 2d strides
        self.pd1, self.pd2, self.pd3, self.pd4 = (0, 0), (0, 0), (0, 0), (0, 0)  # 2d padding

        # encoding components
        resnet = get_pretrained_model(model_name)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet_modules=modules
        self.resnet = nn.Sequential(*modules)
        self.fc1 = nn.Linear(resnet.fc.in_features, self.fc_hidden1)
        self.bn1 = nn.BatchNorm1d(self.fc_hidden1, momentum=0.01)
        self.fc2 = nn.Linear(self.fc_hidden1, self.fc_hidden2)
        self.bn2 = nn.BatchNorm1d(self.fc_hidden2, momentum=0.01)
        # Latent vectors mu and sigma
        self.fc3_mu = nn.Linear(self.fc_hidden2, self.CNN_embed_dim)      # output = CNN embedding latent variables
        self.fc3_logvar = nn.Linear(self.fc_hidden2, self.CNN_embed_dim)  # output = CNN embedding latent variables

        # Sampling vector
        self.fc4 = nn.Linear(self.CNN_embed_dim, self.fc_hidden2)
        self.fc_bn4 = nn.BatchNorm1d(self.fc_hidden2)
        self.fc5 = nn.Linear(self.fc_hidden2, 64 * 8 * 16)
        self.fc_bn5 = nn.BatchNorm1d(64 * 8 * 16)
        self.relu = nn.ReLU(inplace=True)

        # Decoder
        self.convTrans6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=self.k4, stride=self.s4,
                               padding=self.pd4),
            nn.BatchNorm2d(64, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.convTrans7 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=self.k3, stride=self.s3,
                               padding=self.pd3),
            nn.BatchNorm2d(32, momentum=0.01),
            nn.ReLU(inplace=True),
        )

        self.convTrans8 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=self.k2, stride=self.s2,
                               padding=self.pd2),
            nn.BatchNorm2d(16, momentum=0.01),
            nn.ReLU(inplace=True),
     # y = (y1, y2, y3) \in [0 ,1]^3
        )

        self.convTrans9 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=self.k2, stride=self.s2,
                               padding=self.pd2),
            nn.BatchNorm2d(8, momentum=0.01),    # y = (y1, y2, y3) \in [0 ,1]^3
            nn.ReLU(inplace=True),
        )

        self.convTrans10 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8, out_channels=NB_OUTPUT_CHANNELS, kernel_size=self.k2, stride=self.s2,
                               padding=self.pd2),
            nn.BatchNorm2d(NB_OUTPUT_CHANNELS, momentum=0.01),
            Trim(),
            nn.Sigmoid()    # y = (y1, y2, y3) \in [0 ,1]^3
        )



    def encode(self, x):
        x = self.resnet(x)  # ResNet
        x = x.view(x.size(0), -1)  # flatten output of conv

        # FC layers
        x = self.bn1(self.fc1(x))
        x = self.relu(x)
        x = self.bn2(self.fc2(x))
        x = self.relu(x)
        # x = F.dropout(x, p=self.drop_p, training=self.training)
        mu, logvar = self.fc3_mu(x), self.fc3_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        x = self.fc4(z)
        x = self.relu(self.fc_bn4(x))
        x = self.fc5(x)
        x = self.relu(self.fc_bn5(x)).view(-1, 64, 8, 16)
        x = self.convTrans6(x)
        x = self.convTrans7(x)
        x = self.convTrans8(x)
        x = self.convTrans9(x)
        x = self.convTrans10(x)
        # x = F.interpolate(x, size=(270, 360), mode='bilinear')
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_reconst = self.decode(z)
        return x_reconst, z, mu, logvar

class CNN_simple(nn.Module):
    def __init__(self, nbPts = 4,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = torchvision.models.vgg16(pretrained=True)
        self.model.classifier = nn.Sequential(
            nn.Linear(512*7*7,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p = 0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096,nbPts*2)
        )
        # for param in self.model.parameters():
        #     param.requires_grad = False
        # for param in self.model.classifier.parameters():
        #     # print(param.requires_grad)
        #     param.requires_grad = True
        # # self.model.classifier[6] = nn.Linear(4096,nbPts)

    def forward(self, x:torch.tensor):
        x= self.model(x)
        x = x.view(-1,4,2)
        return x


if __name__ == "__main__":
    model = ResNet_VAE(model_name = 'resnet34',NB_OUTPUT_CHANNELS=4).to('cuda')
    model.eval()
    img = torch.rand((16,3,270,360)).to('cuda')
    print(model(img)[0].shape)

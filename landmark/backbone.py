import torch
import torch.nn as nn
from einops import rearrange

from .basenet import MobileNet_GDConv
from .pfld_compressed import PFLDInference
from .mobilefacenet import MobileFaceNet
import os

def load_model(backbone, map_location):
    if backbone=='MobileNet':
        model = MobileNet_GDConv(136)
        # download model from https://drive.google.com/file/d/1Le5UdpMkKOTRr1sTp4lwkw8263sbgdSe/view?usp=sharing
        checkpoint = torch.load(os.path.join(os.path.dirname(__file__),'mobilenet_224_model_best_gdconv_external.pth.tar'), map_location=map_location)
        checkpoint['state_dict'] = {k.replace('module.',''):v for k,v in checkpoint['state_dict'].items()}
        print('Use MobileNet as backbone')
    elif backbone=='PFLD':
        model = PFLDInference() 
        # download from https://drive.google.com/file/d/1gjgtm6qaBQJ_EY7lQfQj3EuMJCVg9lVu/view?usp=sharing
        checkpoint = torch.load(os.path.join(os.path.dirname(
            __file__), 'pfld_model_best.pth.tar'), map_location=map_location)
        print('Use PFLD as backbone') 
        # download from https://drive.google.com/file/d/1T8J73UTcB25BEJ_ObAJczCkyGKW5VaeY/view?usp=sharing
    elif backbone=='MobileFaceNet':
        model = MobileFaceNet([112, 112],136)   
        checkpoint = torch.load(os.path.join(os.path.dirname(__file__), 'mobilefacenet_model_best.pth.tar'), map_location=map_location)
        print('Use MobileFaceNet as backbone')         
    else:
        print('Error: not suppored backbone')   
     
    model.load_state_dict(checkpoint['state_dict'])
    return model


class Landmark(nn.Module):
    def __init__(self, backbone='MobileFaceNet'):
        super(Landmark, self).__init__()
        if torch.cuda.is_available():
            map_location=lambda storage, loc: storage.cuda()
        else:
            map_location='cpu'
            
        self.backbone = load_model(backbone, map_location)
        
        self.size = 112
        if backbone=='MobileNet' or backbone=='MobileNetV2':
            self.size = 224
        elif backbone=='MobileNetV2_56':
            self.size = 56

        for param in self.parameters():
            param.requires_grad = False
            
    @torch.no_grad()
    def forward(self, x, extract_feature=True):
        B, C, T, H, W = x.shape
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        if x.shape[2] != self.size or x.shape[3] != self.size:
            x = nn.functional.interpolate(x, size=(self.size, self.size), mode='bilinear', align_corners=True)
        x, feat = self.backbone(x)
        if extract_feature:
            return feat    
        else:
            return x, feat
        
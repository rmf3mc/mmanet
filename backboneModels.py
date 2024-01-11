import torchvision.models as models

from torchvision.models import DenseNet161_Weights, DenseNet121_Weights , ResNet50_Weights, ResNet34_Weights, ResNet18_Weights,VGG19_Weights, MobileNet_V2_Weights, MobileNet_V3_Large_Weights



import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.nn import TripletMarginLoss
import torchvision
import torchvision.transforms as transforms

from unet3 import *
from utils import *
from torchvision.ops import DeformConv2d




class MMANET(nn.Module):
    def __init__ (self, backbone_name, num_classes,MANet=False,MMANet=True,mask_guided=False,seg_included=None,freeze_all=False,no_sig_classes=1,Unet=True,deform_expan=1):
        super(MMANET, self).__init__()
        
        self.MANet=MANet
        self.MMANet=MMANet
        self.backbone_name=backbone_name
        self.mask_guided=mask_guided
        
        self.Unet=Unet
        self.seg_included= seg_included 
        self.no_classes=no_sig_classes
        self.freeze_all=freeze_all

        self.deform_expan=deform_expan
        
        print('self.seg_included:',self.seg_included)
        
        
        if backbone_name[:-3]=='densenet':          
            self.features=DensenetEncoder(backbone_name, num_classes)         
            self.classifier=self.features.classifier
            self.attention=nn.Conv2d(2,1,kernel_size=1, bias=False)
            if self.freeze_all:
                self._freeze_layers(self.features.features)                
            else:
                self._freeze_layers(self.features.features,upto=9)        
            
     
        elif backbone_name[:-2]=='resnet':
            self.features=ResNetEncoder(backbone_name, num_classes)         
            self.classifier=self.features.classifier
            self.attention=nn.Conv2d(2,1,kernel_size=1, bias=False)
            if self.freeze_all:
                self._freeze_layers(self.features.features)                
            else:
                self._freeze_layers(self.features.features,upto=7)
            self.features.features=nn.Sequential(*list(self.features.features.children())[:-2])


        else:
            self.features=MobileNet(backbone_name,num_classes)
            self.classifier=self.features.classifier
            self.attention=nn.Conv2d(2,1,kernel_size=1, bias=False)
            if self.freeze_all:
                self._freeze_layers(self.features.features)                
            else:
                self._freeze_layers(self.features.features,upto=11)




        if self.seg_included:
        
              
            print('**********************************')
            
            print(len(self.features.features))
            self.Encoders, encoder_mils,no_outputs_ch =set_encoder_layers(self.features.features)                
            
            
            
            shape=no_outputs_ch[-1]

            self.center=nn.Conv2d(int(shape*self.deform_expan+2), shape, kernel_size=3, padding=1).to('cuda')

            self.decoder_layers=nn.ModuleDict()
            
            if self.Unet:
                for i in range(1,6):
                    self.decoder_layers[str(i)]=UNetDecoderLayerModule2(lvl=i,no_channels=no_outputs_ch,no_classes=self.no_classes,deform_expan=self.deform_expan)
                    #self.decoder_layers[str(i)]=UNetDecoderLayerModule(lvl=i,no_channels=no_outputs_ch,no_classes=self.no_classes)
            else:
                for i in range(1,6):
                    self.decoder_layers[str(i)]=UNet3PlusDecoderLayerModule(lvl=i,no_channels=no_outputs_ch,no_classes=self.no_classes)


            self.atten_layers= nn.ModuleDict()
            self.offset_layers=nn.ModuleDict()
            for i in range(1,6):
                self.atten_layers[str(i)]= DeformConv2d(in_channels=no_outputs_ch[i-1], out_channels=int(no_outputs_ch[i-1]*(self.deform_expan-1)), kernel_size=3, padding=1) #nn.Conv2d(2,1,kernel_size=1, bias=False)
                self.offset_layers[str(i)]= nn.Conv2d(in_channels=no_outputs_ch[i-1], out_channels=18, kernel_size=3, padding=1)

   
            
    def _freeze_layers(self, model, upto=False):
        cnt, th = 0, 0
        print('freeze layers:')
        if upto:
            th = upto
            print(f'Freeze layers up to {th}s layer')
        else:
            th=float('inf')
        for name, child in model.named_children():
            cnt += 1
            if cnt < th:
                for name2, params in child.named_parameters():
                    params.requires_grad = False
                    print(name, name2, cnt,params.requires_grad)            
        
        if self.freeze_all:
            for param in self.features.parameters():
                param.requires_grad = False
            for param in self.attention.parameters():
                param.requires_grad = False
            for param in self.classifier.parameters():
                param.requires_grad = False    
        

    def getAttFeats(self,att_map,features):
        features=0.5*features+0.5*(att_map*features)
        return features
    
        
        
        
    def forward(self,x,mask=None):

        features=self.features(x)
            
        outputs={}
        
        if self.MANet:
            fg_att=torch.mean(features,dim=1).unsqueeze(1)   
            fg_att=torch.sigmoid(fg_att)  
            features=self.getAttFeats(fg_att,features)
        
        elif self.MMANet:
            fg_att=self.attention(torch.cat((torch.mean(features,dim=1).unsqueeze(1),torch.max(features,dim=1)[0].unsqueeze(1)),dim=1))
            fg_att=torch.sigmoid(fg_att)
            features=self.getAttFeats(fg_att,features)

        


        if self.backbone_name[:-3]=='densenet':
            features = F.relu(features, inplace=True)
        

        out = F.adaptive_avg_pool2d(features, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
            
        outputs['out']=out    
            
            
        if self.mask_guided:    

            
            h,w = fg_att.shape[2],fg_att.shape[3]
            mask=F.adaptive_avg_pool2d(mask, (h, w))
            fg_att = fg_att.view(fg_att.shape[0],-1)
            mask = mask.view(mask.shape[0],-1)

            mask += 1e-12
            max_elmts=torch.max(mask,dim=1)[0].unsqueeze(1)
            mask = mask/max_elmts
            
            outputs['mask']=mask
            outputs['fg_att']=fg_att
            
            
        if self.seg_included:
            Encoder_outputs = self.get_encoder_ops(x)
            Encoder_5=Encoder_outputs[4]
            Conv_Encoder_5=self.center(Encoder_5)
            outputs['Final_seg'], outputs['decoder_layer_2'], outputs['decoder_layer_3'], outputs['decoder_layer_4'], outputs['decoder_layer_5']=get_segmentation(self.decoder_layers,Encoder_outputs,Conv_Encoder_5)
            outputs['decoder_layer_2'], outputs['decoder_layer_3'], outputs['decoder_layer_4'], outputs['decoder_layer_5'] = torch.mean(outputs['decoder_layer_2'] ,dim=1).unsqueeze(1), torch.mean(outputs['decoder_layer_3'] ,dim=1).unsqueeze(1), torch.mean(outputs['decoder_layer_4'] ,dim=1).unsqueeze(1), torch.mean(outputs['decoder_layer_5'] ,dim=1).unsqueeze(1)
            
            
        return outputs


    def get_encoder_ops(self,x):
        Encoder_outputs=[]

        for i in range(1,6):
            x = self.Encoders[str(i)](x)

            #fg_att=self.atten_layers[str(i)](torch.cat((torch.mean(x,dim=1).unsqueeze(1),torch.max(x,dim=1)[0].unsqueeze(1)),dim=1))
            #fg_att=torch.sigmoid(fg_att)
            #features=self.getAttFeats(fg_att,x)
            fg_att=torch.sigmoid(torch.cat((torch.mean(x,dim=1).unsqueeze(1),torch.max(x,dim=1)[0].unsqueeze(1)),dim=1))
            offset=self.offset_layers[str(i)](x)
            deform=self.atten_layers[str(i)](x,offset)
            features=torch.cat((x,fg_att,deform),dim=1)
            Encoder_outputs.append(features)
    

        return Encoder_outputs


class DensenetEncoder(nn.Module):
    def __init__(self, backbone_name, num_classes):
        super(DensenetEncoder, self).__init__()
        if backbone_name=='densenet161':
            self.features=getattr(models, backbone_name)(weights=DenseNet161_Weights.DEFAULT).features    
        elif backbone_name=='densenet121':
            self.features=getattr(models, backbone_name)(weights=DenseNet121_Weights.DEFAULT).features
        print('last',self.features[-1].num_features)
        self.classifier=nn.Linear(self.features[-1].num_features, num_classes)
                
    def forward(self, x):
        features = self.features(x)
        return features
    
    
    
    
class ResNetEncoder(nn.Module):
    def __init__(self, backbone_name, num_classes):
        super(ResNetEncoder, self).__init__()
        if backbone_name[-2:]=='50':
            self.features = getattr(models, backbone_name)(weights=ResNet50_Weights.DEFAULT)
            encoder_mils,no_features=get_model_specs(nn.Sequential(*list(self.features.children())[:-2]))
        elif backbone_name[-2:]=='34':
            self.features = getattr(models, backbone_name)(weights=ResNet34_Weights.DEFAULT)
            encoder_mils,no_features=get_model_specs(nn.Sequential(*list(self.features.children())[:-2]))
        elif backbone_name[-2:]=='18':
            self.features = getattr(models, backbone_name)(weights=ResNet18_Weights.DEFAULT)
            encoder_mils,no_features=get_model_specs(nn.Sequential(*list(self.features.children())[:-2]))
        
        print('last',no_features[-1])
        self.classifier = nn.Linear(no_features[-1], num_classes)
    def forward(self, x):
        features = self.features(x)
        return features
    
    
class MobileNet(nn.Module):
    def __init__(self, backbone_name, num_classes):
        super(MobileNet, self).__init__()
        if backbone_name[11]=='2':
            self.features=getattr(models,backbone_name)(weights=MobileNet_V2_Weights.DEFAULT).features
            encoder_mils,no_features=get_model_specs(self.features)
        elif backbone_name[11]=='3':
            self.features=getattr(models,backbone_name)(weights=MobileNet_V3_Large_Weights.DEFAULT).features
            encoder_mils,no_features=get_model_specs(self.features)
        print('last',no_features[-1])
        self.classifier=nn.Sequential(
                    nn.Dropout(0.2),
                    nn.Linear(no_features[-1], num_classes))

    def forward(self, x):
        features = self.features(x)
        return features
                











            

    


    
    

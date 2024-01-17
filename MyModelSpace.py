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
from torchvision.ops import DeformConv2d

from unet3 import *
from utils import *
import numpy as np

import nni
from nni.nas.nn.pytorch import LayerChoice, ModelSpace, MutableDropout, MutableLinear, MutableConv2d




class MyModelSpace(ModelSpace):
    def __init__ (self, backbone_name, num_classes,MANet=False,MMANet=True,mask_guided=False,seg_included=None,freeze_all=False,no_sig_classes=1,Unet=True,deform_expan=1):
        super(MyModelSpace, self).__init__()
        
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

            self.center=nn.Conv2d(int(shape*self.deform_expan), shape, kernel_size=3, padding=1).to('cuda')

            self.decoder_layers=nn.ModuleDict()


            deconv_layers_3_out_channels = nni.choice('deconv_layers_3_out_channels', 
                                                      [int(no_outputs_ch[i-1]/32), 
                                                       int(no_outputs_ch[i-1]/16), 
                                                       int(no_outputs_ch[i-1]/8)])

            deconv_layers_5_out_channels = nni.choice('deconv_layers_5_out_channels', 
                                                      [int(no_outputs_ch[i-1]/32), 
                                                       int(no_outputs_ch[i-1]/16), 
                                                       int(no_outputs_ch[i-1]/8)])

            deconv_layers_7_out_channels = nni.choice('deconv_layers_7_out_channels', 
                                                      [int(no_outputs_ch[i-1]/32), 
                                                       int(no_outputs_ch[i-1]/16), 
                                                       int(no_outputs_ch[i-1]/8)])

            atrous_layers_2_out_channels = nni.choice('atrous_layers_2_out_channels', 
                                                      [int(no_outputs_ch[i-1]/32), 
                                                       int(no_outputs_ch[i-1]/16), 
                                                       int(no_outputs_ch[i-1]/8)])

            atrous_layers_3_out_channels = nni.choice('atrous_layers_3_out_channels', 
                                                      [int(no_outputs_ch[i-1]/32), 
                                                       int(no_outputs_ch[i-1]/16), 
                                                       int(no_outputs_ch[i-1]/8)])

            atrous_layers_4_out_channels = nni.choice('atrous_layers_4_out_channels', 
                                                      [int(no_outputs_ch[i-1]/32), 
                                                       int(no_outputs_ch[i-1]/16), 
                                                       int(no_outputs_ch[i-1]/8)])

            atrous_layers_5_out_channels = nni.choice('atrous_layers_5_out_channels', 
                                                      [int(no_outputs_ch[i-1]/32), 
                                                       int(no_outputs_ch[i-1]/16), 
                                                       int(no_outputs_ch[i-1]/8)])


            mean_max_out_channels = nni.choice('mean_max_out_channels', 
                                                      [int(no_outputs_ch[i-1]/32), 
                                                       int(no_outputs_ch[i-1]/16), 
                                                       int(no_outputs_ch[i-1]/8)])




            nni_number_channels= deconv_layers_3_out_channels +deconv_layers_5_out_channels + deconv_layers_7_out_channels + atrous_layers_2_out_channels + atrous_layers_3_out_channels +atrous_layers_4_out_channels +atrous_layers_5_out_channels + mean_max_out_channels

            print(nni_number_channels)
            
            if self.Unet:
                for i in range(1,6):
                    self.decoder_layers[str(i)]=UNetDecoderLayerModule3(lvl=i,no_channels=no_outputs_ch,nni_number_channels=nni_number_channels,no_classes=self.no_classes)
                    #self.decoder_layers[str(i)]=UNetDecoderLayerModule(lvl=i,no_channels=no_outputs_ch,no_classes=self.no_classes)
            else:
                for i in range(1,6):
                    self.decoder_layers[str(i)]=UNet3PlusDecoderLayerModule(lvl=i,no_channels=no_outputs_ch,no_classes=self.no_classes)


            self.deconv_layers_3= nn.ModuleDict()
            self.deconv_layers_5= nn.ModuleDict()
            self.deconv_layers_7= nn.ModuleDict()


            self.atrous_conv_layers_2=nn.ModuleDict()
            self.atrous_conv_layers_3=nn.ModuleDict()
            self.atrous_conv_layers_4=nn.ModuleDict()
            self.atrous_conv_layers_5=nn.ModuleDict()


            self.adaptive_layers_3=nn.ModuleDict()
            self.adaptive_layers_5=nn.ModuleDict()

            self.max_min_expan_layers=nn.ModuleDict()



            
            for i in range(1,6):
                all_out_channels=[int(no_outputs_ch[i-1]*(deform_expan-1)/8) for _ in range(7) ]
                max_mean_layer_outchannels = int(no_outputs_ch[i-1]*(deform_expan-1) - np.sum(all_out_channels))


                self.deconv_layers_3[str(i)]= Deform_Conv(in_channels=no_outputs_ch[i-1], out_channels=all_out_channels[0], kernel_size=3)
                self.deconv_layers_5[str(i)]= Deform_Conv(in_channels=no_outputs_ch[i-1], out_channels=all_out_channels[1], kernel_size=5)
                self.deconv_layers_7[str(i)]= Deform_Conv(in_channels=no_outputs_ch[i-1], out_channels=all_out_channels[2], kernel_size=7) 



                self.atrous_conv_layers_2[str(i)]= nn.Conv2d(in_channels=no_outputs_ch[i-1], out_channels=all_out_channels[3], kernel_size=3, dilation=2, padding=2)
                self.atrous_conv_layers_3[str(i)]= nn.Conv2d(in_channels=no_outputs_ch[i-1], out_channels=all_out_channels[4], kernel_size=3, dilation=3, padding=3)
                self.atrous_conv_layers_4[str(i)]= nn.Conv2d(in_channels=no_outputs_ch[i-1], out_channels=all_out_channels[5], kernel_size=3, dilation=4, padding=4)
                self.atrous_conv_layers_5[str(i)]= nn.Conv2d(in_channels=no_outputs_ch[i-1], out_channels=all_out_channels[6], kernel_size=3, dilation=5, padding=5)




                self.deconv_layers_3[str(i)] = Deform_Conv(in_channels=no_outputs_ch[i-1], 
                                                           out_channels=deconv_layers_3_out_channels, 
                                                           kernel_size=3)

                self.deconv_layers_5[str(i)] = Deform_Conv(in_channels=no_outputs_ch[i-1], 
                                                           out_channels=deconv_layers_5_out_channels, 
                                                           kernel_size=5)

                self.deconv_layers_7[str(i)] = Deform_Conv(in_channels=no_outputs_ch[i-1], 
                                                           out_channels=deconv_layers_7_out_channels, 
                                                           kernel_size=7)

                self.atrous_conv_layers_2[str(i)] = nn.Conv2d(in_channels=no_outputs_ch[i-1], 
                                                              out_channels=atrous_layers_2_out_channels, 
                                                              kernel_size=3, dilation=2, padding=2)

                self.atrous_conv_layers_3[str(i)] = nn.Conv2d(in_channels=no_outputs_ch[i-1], 
                                                              out_channels=atrous_layers_3_out_channels, 
                                                              kernel_size=3, dilation=3, padding=3)

                self.atrous_conv_layers_4[str(i)] = nn.Conv2d(in_channels=no_outputs_ch[i-1], 
                                                              out_channels=atrous_layers_4_out_channels, 
                                                              kernel_size=3, dilation=4, padding=4)

                self.atrous_conv_layers_5[str(i)] = nn.Conv2d(in_channels=no_outputs_ch[i-1], 
                                                              out_channels=atrous_layers_5_out_channels, 
                                                              kernel_size=3, dilation=5, padding=5)






                self.max_min_expan_layers[str(i)]= nn.Conv2d(2,out_channels=mean_max_out_channels,kernel_size=1)



   
            
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
                    #print(name, name2, cnt,params.requires_grad)            
        
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

            mean_max=torch.cat((torch.mean(x,dim=1).unsqueeze(1),torch.max(x,dim=1)[0].unsqueeze(1)),dim=1)
        
        
            deform_3=self.deconv_layers_3[str(i)](x)
            deform_5=self.deconv_layers_5[str(i)](x)
            deform_7=self.deconv_layers_7[str(i)](x)


            atrous_2=self.atrous_conv_layers_2[str(i)](x)
            atrous_3=self.atrous_conv_layers_3[str(i)](x)
            atrous_4=self.atrous_conv_layers_4[str(i)](x)
            atrous_5=self.atrous_conv_layers_5[str(i)](x)


            # adapt_3=self.adaptive_layers_3[str(i)](x)
            # adapt_5=self.adaptive_layers_5[str(i)](x)

            mean_max_expan=self.max_min_expan_layers[str(i)](mean_max)

            features=torch.cat((x,deform_3,deform_5,deform_7,atrous_2,atrous_3,atrous_4,atrous_5,mean_max_expan),dim=1)
            Encoder_outputs.append(features)
            #print(f'Layer no:{i}, input features:{x.shape}, out number of features: {features.shape}')
    


        return Encoder_outputs


class Deform_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Deform_Conv, self).__init__()
        self.offset_generator = nn.Sequential(
            nn.Conv2d(in_channels, 64, 1),
            nn.ReLU(),
            nn.Conv2d(64,  2 * kernel_size * kernel_size, 1)
        )
        padding=kernel_size//2
        self.deform_conv = DeformConv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        

    def forward(self, x):
        offset = self.offset_generator(x)
        out = self.deform_conv(x, offset)
        return out




class SpatiallyAdaptiveConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(SpatiallyAdaptiveConv, self).__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Standard Convolution
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)

        # Adaptive network to modify the weights
        self.adapt_net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 1),
            nn.ReLU(),
            nn.Conv2d(64, out_channels * in_channels * kernel_size * kernel_size, 1)
        )

    def forward(self, x):
        # Generate adaptive weights
        adaptive_weights = self.adapt_net(x)

        # Reshape adaptive weights to match the convolution weights shape
        adaptive_weights = adaptive_weights.view(-1, self.in_channels, self.kernel_size, self.kernel_size)

        # Apply adaptive convolution
        output = F.conv2d(x, adaptive_weights, padding=self.kernel_size // 2)
        return output



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
                











            

    


    
    

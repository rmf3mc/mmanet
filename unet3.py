import torch
import torch.nn as nn
from nni.nas.nn.pytorch import LayerChoice, ModelSpace, MutableDropout, MutableLinear, MutableConv2d

class UNet3PlusDecoderLayerModule(nn.Module):
    def __init__(self, lvl,no_channels,no_classes):
        super(UNet3PlusDecoderLayerModule, self).__init__()
        self.layers= nn.ModuleDict()
        if lvl==1:
            self.decoder_layer_1=True
        else:
            self.decoder_layer_1=False
        for i in range(1,6):
            SF=self.determine_updo_scaling (i,lvl)
            in_channels=no_channels[i-1]
            #out_channels=no_channels[lvl-1]
            if i ==lvl+1:
                self.layers[str(i)]=self.conv_block( in_channels=no_classes, out_channels=64,SF=SF)
            else:
                self.layers[str(i)]=self.conv_block( in_channels=in_channels, out_channels=64,SF=SF)
                
        self.layers[str(6)]=self.conv_block( in_channels=320, out_channels=no_classes,SF=1,Final=True)

        
        
    def conv_block(self, in_channels, out_channels,SF,Final=False):
        if SF>1:
            SF=int(SF)
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.Upsample(scale_factor=SF, mode='bilinear', align_corners=True)
                )
        elif SF<1:
            stride=int(SF**-1)
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.MaxPool2d(3, stride=stride,padding=1)
                )
        
        else:
            if Final and not (self.decoder_layer_1):
                return nn.Sequential( nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                       nn.BatchNorm2d(out_channels),  
                       nn.ReLU(inplace=True)                   
                       ) 
            else:
                return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)


                    
        
    def determine_updo_scaling (self,From,To):
        return 2**(From-To)
    
    
    
    def forward (self,Enc_outputs,next_decoder_layer_output,lvl):
          
            

            outputs=[self.layers[str(i)](next_decoder_layer_output) if i ==lvl+1 else self.layers[str(i)](Enc_outputs[i-1]) for i in range(1,6) ]


            concatenated_output = torch.cat(outputs, dim=1)
            final_output=self.layers[str(6)](concatenated_output)


            return final_output





def get_segmentation(decoder_layers,Encoder_outputs,Conv_Encoder_5):

    # Now Encoder_outputs contains the output of each layer
    #decoder_outputs = [self.feed_decoders(Encoder_outputs,self.decoder_layers[i]) for i in range(1,6)]
    
    
    #     for f in range(5):
    #         print(f'Encoder_outputs[{f}].shap',Encoder_outputs[f].shape)
        
    #print('Conv_Encoder_5.shape',Conv_Encoder_5.shape)
    
    i=5
    decoder_output_5= decoder_layers[str(i)](Encoder_outputs[i-1],Conv_Encoder_5)
    #print('decoder_output_5.shape',decoder_output_5.shape)
    
    #print(Encoder_outputs[3].shape,decoder_output_5.shape)
    i=4
    decoder_output_4= decoder_layers[str(i)](Encoder_outputs[i-1],decoder_output_5) 
              
    #print('decoder_output_4.shape',decoder_output_4.shape)
    
    i=3
    decoder_output_3= decoder_layers[str(i)](Encoder_outputs[i-1],decoder_output_4)
    #print('decoder_output_3.shape',decoder_output_3.shape)
    
    i=2
    decoder_output_2= decoder_layers[str(i)](Encoder_outputs[i-1],decoder_output_3) 
              
    #print('decoder_output_2.shape',decoder_output_2.shape)
    
    i=1
    Final_seg= decoder_layers[str(i)](Encoder_outputs[i-1],decoder_output_2)
    #print('Final_seg',Final_seg.shape)
    
    return Final_seg,decoder_output_2,decoder_output_3,decoder_output_4,decoder_output_5





class UNetDecoderLayerModule(nn.Module):
    def __init__(self, lvl,no_channels,no_classes=1):
        super(UNetDecoderLayerModule, self).__init__()
        self.layers= nn.ModuleDict()
        in_channels=no_channels[lvl-1]*2
        if lvl==1:
            out_channels=no_channels[lvl-1]
        else:
            out_channels=no_channels[lvl-2]
            
        print('out_channels',out_channels)
        if lvl !=1:
            self.layers[str(1)]=nn.Sequential(
                                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(out_channels, out_channels=out_channels, kernel_size=3, padding=1),
                                nn.ReLU(inplace=True),
                                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                                )
        else:
            self.layers[str(1)]=nn.Sequential(
                        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(out_channels, out_channels=out_channels, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(out_channels, out_channels=no_classes, kernel_size=3, padding=1),
                        )
        
    def forward (self,Enc_outputs,next_decoder_layer_output,lvl):
        concat=torch.cat([Enc_outputs[lvl-1], next_decoder_layer_output], 1)
        out=self.layers[str(1)](concat)
        return out
    


class UNetDecoderLayerModule2(nn.Module):
    def __init__(self, lvl,no_channels,no_classes=1,deform_expan=1):
        super(UNetDecoderLayerModule2, self).__init__()
        self.layers= nn.ModuleDict()
        in_channels=no_channels[lvl-1]#*2
        if lvl==1:
            out_channels=no_channels[lvl-1]
        else:
            out_channels=no_channels[lvl-2]
            
        print('out_channels',out_channels)

        if lvl !=1:
            self.layers[str(1)]=nn.Sequential(
                                nn.Conv2d(in_channels=int(in_channels*(deform_expan+1)), out_channels=out_channels, kernel_size=3, padding=1),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(out_channels, out_channels=out_channels, kernel_size=3, padding=1),
                                nn.ReLU(inplace=True),
                                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                                )
        else:
            self.layers[str(1)]=nn.Sequential(
                        nn.Conv2d(in_channels=int(in_channels*(deform_expan+1)), out_channels=out_channels, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(out_channels, out_channels=out_channels, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(out_channels, out_channels=no_classes, kernel_size=3, padding=1),
                        )
            
        
    def forward (self,Enc_output,next_decoder_layer_output):
        concat=torch.cat([Enc_output, next_decoder_layer_output], 1)
        out=self.layers[str(1)](concat)
        return out
        





def get_model_specs(model,print_feat=False):
    if  next(model.parameters()).is_cuda:
        x = torch.randn(1, 3, 224, 224).to('cuda')
    else:
        x = torch.randn(1, 3, 224, 224)

    shape_prev=x.shape[-1]
    encoder_mils=[]
    all_features_shape=[]
    no_features=[]
    for i in range(len(model)):
        x=model[i](x)
        all_features_shape.append(x.shape[1])
        if print_feat:
            print(i,x.shape)
        if x.shape[-1] != shape_prev:
            encoder_mils.append(i)
            shape_prev=x.shape[-1]
    print('************************')
    for i in range(len(encoder_mils)):
        no_features.append(all_features_shape[encoder_mils[i]-1])
    
    du_variable=no_features[1:]+no_features[:1]
    no_features=du_variable
    return encoder_mils,no_features



def set_encoder_layers(model):  
    encoder_mils,no_outputs_ch= get_model_specs(model,print_feat=True)
    print('Down sample at',encoder_mils)
    print('Number of out channels', no_outputs_ch)
    layers=nn.ModuleDict()
    leng=len(encoder_mils)
    for i in range(0,leng):
        if i!=leng-1:
            print(f'From_Layer:{encoder_mils[i]} to_Layer:{encoder_mils[i+1]-1}')
            layers[str(i+1)]=model[encoder_mils[i]:encoder_mils[i+1]]
        else:
            print(f'From_Layer:{encoder_mils[i]} to_End')
            layers[str(i+1)]=model[encoder_mils[i]:]
    print('********************')
    return layers, encoder_mils, no_outputs_ch




def get_no_output(model,layer_depth=0):  
    num_children = sum(1 for _ in model.children())
    i=0
    first_op=0 
    for child in model.children():
        i+=1
        #print("  " * layer_depth , child.__class__.__name__)
        if  child.__class__.__name__ == 'Conv2d':
            #print("  " *layer_depth , child.__class__.__name__,child.in_channels, child.out_channels)
            return child.in_channels
        if list(child.children()):
            first_op=get_no_output(child, layer_depth + 1)     
            if first_op!=0:
                return first_op

def find_latest_batchnorm(encoder_module):
    for module in reversed(list(encoder_module.modules())):
        if isinstance(module, nn.BatchNorm2d):  # or nn.BatchNorm1d, nn.BatchNorm3d based on your use case
            return module.num_features
    return None  # Return None if no BatchNorm layer is found



# def get_segmentation(decoder_layers,Encoder_outputs):

#     # Now Encoder_outputs contains the output of each layer
#     #decoder_outputs = [self.feed_decoders(Encoder_outputs,self.decoder_layers[i]) for i in range(1,6)]
#     i=5
#     decoder_output_5= decoder_layers[str(i)](Encoder_outputs,None,i)  #self.feed_decoders(Encoder_outputs,self.decoder_layers[str(i)],None,i)
#     i=4
#     decoder_output_4= decoder_layers[str(i)](Encoder_outputs,decoder_output_5,i) #self.feed_decoders(Encoder_outputs,self.decoder_layers[str(i)],decoder_output_5,i)
#     i=3
#     decoder_output_3= decoder_layers[str(i)](Encoder_outputs,decoder_output_4,i) #self.feed_decoders(Encoder_outputs,self.decoder_layers[str(i)],decoder_output_4,i)
#     i=2
#     decoder_output_2= decoder_layers[str(i)](Encoder_outputs,decoder_output_3,i) #self.feed_decoders(Encoder_outputs,self.decoder_layers[str(i)],decoder_output_3,i)   
#     i=1
#     Final_seg= decoder_layers[str(i)](Encoder_outputs,decoder_output_2,i) #self.feed_decoders(Encoder_outputs,self.decoder_layers[str(i)],decoder_output_2,i)
#     return Final_seg
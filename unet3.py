import torch
import torch.nn as nn


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
    decoder_output_5= decoder_layers[str(i)](Encoder_outputs,Conv_Encoder_5,i)
    #print('decoder_output_5.shape',decoder_output_5.shape)
    
    #print(Encoder_outputs[3].shape,decoder_output_5.shape)
    i=4
    decoder_output_4= decoder_layers[str(i)](Encoder_outputs,decoder_output_5,i) 
              
    #print('decoder_output_4.shape',decoder_output_4.shape)
    
    i=3
    decoder_output_3= decoder_layers[str(i)](Encoder_outputs,decoder_output_4,i)
    #print('decoder_output_3.shape',decoder_output_3.shape)
    
    i=2
    decoder_output_2= decoder_layers[str(i)](Encoder_outputs,decoder_output_3,i) 
              
    #print('decoder_output_2.shape',decoder_output_2.shape)
    
    i=1
    Final_seg= decoder_layers[str(i)](Encoder_outputs,decoder_output_2,i)
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
    def __init__(self, lvl,no_channels,no_classes=1,att_fromm=1):
        super(UNetDecoderLayerModule2, self).__init__()
        self.layers= nn.ModuleDict()
        in_channels=no_channels[lvl-1]#*2
        if lvl==1:
            out_channels=no_channels[lvl-1]
        else:
            out_channels=no_channels[lvl-2]
            
        print('out_channels',out_channels)
        if lvl<att_fromm:
            if lvl !=1:
                self.layers[str(1)]=nn.Sequential(
                                    nn.Conv2d(in_channels=in_channels+in_channels, out_channels=out_channels, kernel_size=3, padding=1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(out_channels, out_channels=out_channels, kernel_size=3, padding=1),
                                    nn.ReLU(inplace=True),
                                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                                    )
            else:
                self.layers[str(1)]=nn.Sequential(
                            nn.Conv2d(in_channels=in_channels+in_channels, out_channels=out_channels, kernel_size=3, padding=1),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(out_channels, out_channels=out_channels, kernel_size=3, padding=1),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(out_channels, out_channels=no_classes, kernel_size=3, padding=1),
                            )
        else:
            if lvl !=1:
                self.layers[str(1)]=nn.Sequential(
                                    nn.Conv2d(in_channels=in_channels+int(in_channels*1.25)+2, out_channels=out_channels, kernel_size=3, padding=1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(out_channels, out_channels=out_channels, kernel_size=3, padding=1),
                                    nn.ReLU(inplace=True),
                                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                                    )
            else:
                self.layers[str(1)]=nn.Sequential(
                            nn.Conv2d(in_channels=in_channels+int(in_channels*1.25)+2, out_channels=out_channels, kernel_size=3, padding=1),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(out_channels, out_channels=out_channels, kernel_size=3, padding=1),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(out_channels, out_channels=no_classes, kernel_size=3, padding=1),
                            )
            
        
    def forward (self,Enc_outputs,next_decoder_layer_output,lvl):
        concat=torch.cat([Enc_outputs[lvl-1], next_decoder_layer_output], 1)
        out=self.layers[str(1)](concat)
        return out
        




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
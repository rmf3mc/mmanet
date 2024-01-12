import torch
import torch.nn as nn

def get_folder_path(args):
    
    path_parts = []

    # Append the name of the argument to the list if it's True
    
    path_parts.append(args.dataset[:3])


    if args.seg_ild:
        path_parts.append(str(args.deform_expan))
        if args.unet:
            path_parts.append('Unet')
        else:
            path_parts.append('Unet3Plus')
        
        if args.fsds:
            path_parts.append('FSDS')
        
        
    
    path_parts.append(args.backbone_class)
    
    if args.manet:
        path_parts.append('MANet')

    elif args.mmanet:
        path_parts.append('MMANet')
        
    else:
        path_parts.append('Original')
        
        
    if args.maskguided:
        path_parts.append('MaskGuided')
              

    if args.cls_ild:
        path_parts.append('Classification')
        
                

            
        

    # Join the parts together with underscores
    args_part = '_'.join(path_parts)
    
    
    return args_part
    
    
    
def iou_binary(preds, labels, EMPTY=1e-9):
    """
    Calculate Intersection over Union (IoU) for a single pair of binary segmentation masks using bitwise operations.
    
    Parameters:
    - preds (Tensor): Predicted segmentation mask, shape [1, height, width]
    - labels (Tensor): Ground truth segmentation mask, shape [1, height, width]
    - EMPTY (float): A small constant to prevent division by zero
    
    Returns:
    - IoU (float): Intersection over Union score
    """
    
    # Convert to boolean tensors
    preds = preds.bool()
    labels = labels.bool()

    # Calculate intersection and union using bitwise operations
    intersection = (preds & labels).float().sum()
    union = (preds | labels).float().sum()

    # Calculate IoU
    IoU = (intersection + EMPTY) / (union + EMPTY)

    return IoU.item()


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
import torch
import torch.nn as nn

def get_folder_path(args):
    
    path_parts = []

    # Append the name of the argument to the list if it's True
    



    if args.seg_ild:
        path_parts.append(str(args.deform_expan))
        if args.unet:
            path_parts.append('Unet')
        else:
            path_parts.append('Unet3Plus')
        
        if args.fsds:
            path_parts.append('FSDS')
        
        
    path_parts.append(args.dataset[:3])
        
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



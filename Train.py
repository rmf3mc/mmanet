#!/usr/bin/env python
# coding: utf-8

# In[1]:


print('**Code Starting**')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from   torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

import numpy as np
import pickle
import random
import os
import argparse

import time
from datetime import datetime

from leafvein2 import Leafvein

from backboneModels import *
from utils import *

print('**End of Importing**')


# In[2]:


from torchvision.ops import DeformConv2d


# In[ ]:


## reproducility
seed=1
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.cuda.manual_seed_all(seed)
cudnn.benchmark = False 


# In[2]:


test_acc=[]
train_acc=[]
train_total_losses=[]
test_total_losses=[]
train_iou=[]
test_iou=[]


# In[12]:


import argparse

# Create the parser
parser = argparse.ArgumentParser(description='Training PyTorch Models For Ultra Fine Grained')

# Add arguments
parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--max_epoch', default=150, type=int)
parser.add_argument('--backbone_class', type=str, default='densenet161', choices=['densenet161',  'resnet50', 'resnet34', 'resnet18', 'mobilenet_v2', 'mobilenet_v3_large' ], help='Backbone models')
parser.add_argument('--dataset', type=str, default='soybean_2_1', choices=['soybean_2_1', 'btf', 'hainan_leaf'], help='resume from checkpoint')
parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('--seg_size', default=448, type=int, help='Segmentation Dimension')
parser.add_argument('--num_classes', default=200, type=int, help='Number of Classes')

parser.add_argument('--dataparallel', action='store_true', help='Enable Data Parallel')
parser.add_argument('--seg_ild', action='store_true', help='Enable Segmentation Training')
parser.add_argument('--cls_ild', action='store_true', help='Enable Classification Training')
parser.add_argument('--freeze_all', action='store_true', help='Freeze the Encoder Module')
parser.add_argument('--manet', action='store_true', help='Using the MANet')
parser.add_argument('--mmanet', action='store_true', help='Using the MMANet')
parser.add_argument('--maskguided', action='store_true',help='Guiding the Attention Mask')
parser.add_argument('--unet', action='store_true', help='Unet based Segmentation, Unet3+ otherwise')
parser.add_argument('--deform_expan', default=3,type=float, help='Applying mean attention to Encoder outputs')

parser.add_argument('--model_path', type=str, help='The pretrained model path')
parser.add_argument('--fsds', action='store_true', help='Using Full-scale Deep Supervision')

parser.add_argument('--local_train', default= 0 , type=int, help='local_training')



# Use parse_known_args()
args, unknown = parser.parse_known_args()

print('args.local_train',args.local_train)


if args.model_path is not None:
    args.batch_size=8
    if not args.unet:
        args.lr=0.02
else:
    args.batch_size=32


# In[5]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device:',device)


# In[6]:


batchsize = args.batch_size


MANet=args.manet
MMANet=args.mmanet
mask_guided=args.maskguided

seg_ild=args.seg_ild
cls_ild=args.cls_ild
freeze_all=args.freeze_all

num_classes=args.num_classes

model_name=args.backbone_class

start_epoch=0


# In[4]:


#current_working_directory = os.getcwd()
if args.local_train==1:
    current_working_directory = '/home/pupil/rmf3mc/Documents/ModelProposing/MGANet/FinalTouches_AMP/'
else:
    current_working_directory = '/mnt/mywork/all_backbones/'
print("Current Working Directory:", current_working_directory)

name=get_folder_path(args)
print('folder_path',name)


# Get the current date and time
current_datetime = datetime.now()

# Format the hour:minutes date and time as day-month-year 
formatted_datetime = current_datetime.strftime("%H:%M-%d-%m-%Y")

if cls_ild and not(seg_ild):
    train_type=args.dataset+'-results-cls'

elif seg_ild and not (cls_ild):
    if args.unet:
        train_type=os.path.join(args.dataset+'-results-seg','Unet'+str(args.deform_expan))
    else:
        train_type=os.path.join(args.dataset+'-results-seg','Unet3Plus'+str(args.deform_expan))

else:
    if args.unet:
        train_type=os.path.join(args.dataset+'-results-seg-cls','Unet'+str(args.deform_expan))
    else:
        train_type=os.path.join(args.dataset+'-results-seg-cls','Unet3Plus'+str(args.deform_expan))

folder_path =os.path.join(current_working_directory,train_type,args.backbone_class,name,formatted_datetime)
accuracy_file_path =os.path.join(folder_path,'model_accuracies_iou.txt')
    
    
print('File path:',accuracy_file_path)
    
    
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    

models_folder=current_working_directory+'/checkpoint'
if not os.path.exists(models_folder):
    os.makedirs(models_folder)


# In[7]:


print('\nBatch_size',args.batch_size)

print('MANet',MANet)
print('MMANet',MMANet)
print('mask_guided',mask_guided)

print('seg_included',seg_ild)
print('cls_included',cls_ild)

print('freeze_all',freeze_all)
print('Full-scale Deep Supervision',args.fsds)
print('Unet',args.unet)
print('deform_expan',args.deform_expan)


# In[ ]:


start=time.time()
train = Leafvein(args,crop=[448,448],hflip=True,vflip=False,erase=False,mode='train')
test = Leafvein(args,crop=[448,448],mode='test')
end=time.time()
print(end-start)


# In[ ]:


trainloader = DataLoader(train, batch_size=batchsize, shuffle=True)
testloader = DataLoader(test, batch_size=batchsize, shuffle=False)


# In[ ]:


model=MMANET(backbone_name=model_name,num_classes=num_classes,MANet=MANet,MMANet=MMANet,mask_guided=mask_guided,seg_included=seg_ild,freeze_all=freeze_all,Unet=args.unet,deform_expan=args.deform_expan)


if args.model_path is not None:
    print('Loading weights')
    model_path= args.model_path
    model_dict = torch.load(model_path)
    state_dict = model_dict['net'] 
    model.load_state_dict(state_dict, strict=False)
else:
    model_path= f'{current_working_directory}/checkpoint/{name}_{formatted_datetime}.pth'

    
print('model.seg_included',model.seg_included)
print('model.MMANet',model.MMANet)
print('model.MANet',model.MANet)    



net=model.to(device)

if device == 'cuda' and args.dataparallel:
    net = torch.nn.DataParallel(net)

if args.seg_ild:
    model_path= f'{current_working_directory}/checkpoint/{name}-Segmentation-{formatted_datetime}.pth'


# In[ ]:


total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters in the model: {total_params/1e+6}")


with open(accuracy_file_path, 'a') as f:
    f.write(f'\n ***************************Start******************************** \n')
    for arg in vars(args):
        f.write(f"{arg}: {getattr(args, arg)}  \n")
    f.write(f"Total parameters in the model: {total_params/1e+6}")


# In[ ]:


alpha1,alpha2,alpha3,alpha4,alpha5=0.95, 0.1/4, 0.1/8, 0.1/16, 0.1/32

class_loss_fn = nn.CrossEntropyLoss()
seg_loss_fn   = nn.BCEWithLogitsLoss()
mse = nn.MSELoss()




optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch)


# In[ ]:


input_size = [args.seg_size, args.seg_size]
new_size = [size // 2 for size in input_size]

def train_epoch_Seg(epoch):
    print('\nEpoch: %d' % epoch)
    
    global new_size
    with open(accuracy_file_path, 'a') as f:
        f.write(f'\n Epoch:{epoch}\n')

    
    
    net.train()
    if freeze_all and not(cls_ild):
    
        for name, module in net.module.features.named_modules():
            if isinstance(module, (nn.BatchNorm2d, nn.Dropout)):
                module.eval()
                
                
    train_loss = 0
    train_ce_loss=0
    averageIoU=0
    correct = 0
    total = 0


        

    for batch_idx, (inputs, targets, masks) in enumerate(trainloader):
        inputs, targets, masks = inputs.to(device), targets.to(device), masks.to(device)

        masks=masks.unsqueeze(1)
        masks = F.interpolate(masks, size=new_size, mode='bilinear', align_corners=False)
            
        
        optimizer.zero_grad()
     
        ce_loss_,se_loss_,mse_loss_=0,0,0
        
        
        if mask_guided:
            outputs = net(inputs, masks) 
            msks=outputs['mask']
            fg_att=outputs['fg_att']
            mse_loss_ = mse(fg_att,msks)

        else:
            outputs = net(inputs)

        if seg_ild:
            Final_seg=outputs['Final_seg']
            se_loss_ = seg_loss_fn(Final_seg,masks)

            preds = torch.sigmoid(Final_seg)
            preds = (preds > 0.5).float()

            iou = iou_binary(preds, masks)
            averageIoU+=iou

            if args.fsds:


                lvl_2_loss = seg_loss_fn(outputs['decoder_layer_2'],masks)

                fsds_size = [size // 2 for size in new_size]
                masks = F.interpolate(masks, size=fsds_size, mode='bilinear', align_corners=False)
                lvl_3_loss = seg_loss_fn(outputs['decoder_layer_3'],masks)

                fsds_size = [size // 2 for size in fsds_size]
                masks = F.interpolate(masks, size=fsds_size, mode='bilinear', align_corners=False)
                lvl_4_loss = seg_loss_fn(outputs['decoder_layer_4'],masks)

                fsds_size = [size // 2 for size in fsds_size]
                masks = F.interpolate(masks, size=fsds_size, mode='bilinear', align_corners=False)
                lvl_5_loss = seg_loss_fn(outputs['decoder_layer_5'],masks)
                se_loss_= alpha1*se_loss_+ alpha2*lvl_2_loss+ alpha3*lvl_3_loss+ alpha4*lvl_4_loss + alpha5*lvl_5_loss



            out=outputs['out']
            ce_loss_ = class_loss_fn(out, targets)

            loss =  seg_ild*se_loss_ + cls_ild*ce_loss_+ mask_guided*0.1*mse_loss_

            
            
            loss.backward()
            optimizer.step()
        
        

        train_loss += loss.item()
        train_ce_loss+=ce_loss_.item()

        
        _, predicted = out.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        

        
        Final_seg,masks,inputs,targets,outputs=None,None,None,None,None
    
    train_ce_loss/=(batch_idx+1)
    train_loss/=(batch_idx+1)
        
    averageIoU=averageIoU*100/(batch_idx+1)
    accuracy=100.*correct/total
    
    print(f'Acc: {accuracy:.3f}% ({correct}/{total})| CE: {train_ce_loss:.4f}| Total Loss: {train_loss:.4f}| IoU :{averageIoU:.4f}')
    with open(accuracy_file_path, 'a') as f:
        f.write(f'Acc: {accuracy:.3f}% ({correct}/{total})| CE: {train_ce_loss:.4f}| Total Loss: {train_loss:.4f}| IoU :{averageIoU:.4f}\n')

    
    
    train_total_losses.append(train_loss)
    train_iou.append(averageIoU)
    return averageIoU,accuracy,train_ce_loss


# In[ ]:


def test_epoch_Seg(epoch):
    net.eval()
    global best_iou, best_acc, new_size
    test_loss = 0
    test_ce_loss = 0
    
    averageIoU=0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets,masks) in enumerate(testloader):
            inputs, targets,masks= inputs.to(device), targets.to(device),masks.to(device)
            
            
            masks=masks.unsqueeze(1)
            masks = F.interpolate(masks, size=new_size, mode='bilinear', align_corners=False)
        

            ce_loss_,se_loss_,mse_loss_=0,0,0

            if mask_guided:
                outputs=net(inputs,masks)
                msks=outputs['mask']
                fg_att=outputs['fg_att']
                mse_loss_ = mse(fg_att,msks)
            else:
                outputs=net(inputs)
                

            if seg_ild:
                Final_seg=outputs['Final_seg']
                se_loss_ = seg_loss_fn(Final_seg,masks)

                preds = torch.sigmoid(Final_seg)
                preds = (preds > 0.5).float()

                iou = iou_binary(preds, masks)
                averageIoU+=iou
                
                                    
                if args.fsds:
                    
                    
                    lvl_2_loss = seg_loss_fn(outputs['decoder_layer_2'],masks)

                    fsds_size = [size // 2 for size in new_size]
                    masks = F.interpolate(masks, size=fsds_size, mode='bilinear', align_corners=False)
                    lvl_3_loss = seg_loss_fn(outputs['decoder_layer_3'],masks)

                    fsds_size = [size // 2 for size in fsds_size]
                    masks = F.interpolate(masks, size=fsds_size, mode='bilinear', align_corners=False)
                    lvl_4_loss = seg_loss_fn(outputs['decoder_layer_4'],masks)

                    fsds_size = [size // 2 for size in fsds_size]
                    masks = F.interpolate(masks, size=fsds_size, mode='bilinear', align_corners=False)
                    lvl_5_loss = seg_loss_fn(outputs['decoder_layer_5'],masks)
                    se_loss_= alpha1*se_loss_+ alpha2*lvl_2_loss+ alpha3*lvl_3_loss+ alpha4*lvl_4_loss + alpha5*lvl_5_loss


            out=outputs['out']
            ce_loss_ = class_loss_fn(out, targets)

            loss =  seg_ild*se_loss_ + cls_ild*ce_loss_+ mask_guided*0.1*mse_loss_



            test_ce_loss+=ce_loss_.item()
            test_loss += loss.item()
            
            
            _, predicted = out.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            
            
            
           
            segs,masks,inputs,targets,outputs=None,None,None,None,None

    averageIoU=averageIoU*100/(batch_idx+1)
    accuracy=100.*correct/total
    

    if averageIoU>best_iou:
        best_iou=averageIoU
        
    if accuracy>best_acc:
        best_acc=accuracy
    
    test_ce_loss/=(batch_idx+1)
    test_loss/=(batch_idx+1)
    
    test_total_losses.append(test_loss)
    test_iou.append(averageIoU)
    
    print(f'Acc: {accuracy:.3f}% ({correct}/{total})| CE: {test_ce_loss:.4f}|  Total Loss: {test_loss:.4f}| IoU :{averageIoU:.4f}')
    print('cur_iou:{0},best_iou:{1}:'.format(averageIoU,best_iou))
    print('curr_Acc:{0},best_Acc:{1}:'.format(accuracy,best_acc))
    
    
    with open(accuracy_file_path, 'a') as f:
        f.write(f'Acc: {accuracy:.3f}% ({correct}/{total})| CE: {test_ce_loss:.4f}| Total Loss: {test_loss:.4f}| IoU :{averageIoU:.4f}\n')
        f.write(f'cur_iou:{averageIoU},best_iou:{best_iou}\n')
        f.write(f'curr_Acc:{accuracy},best_Acc:{best_acc}\n')

        

        
        

    return averageIoU,accuracy,test_ce_loss


# In[ ]:


def check_seg_performance(iou,curr_test_iou,test_acc,net,epoch,end,start,model_path,accuracy_file_path):
    global train_best_iou,best_test_iou,Best_test_acc_BOT
    if iou>train_best_iou:
        print('Saving..')
        train_best_iou=iou
        best_test_iou=curr_test_iou
        Best_test_acc_BOT=test_acc
        if isinstance(net, torch.nn.DataParallel):
            net_wrap = net.module
        else:
            net_wrap=net
        state = {'net': net_wrap.state_dict(), 'test_iou': best_test_iou, 'Test_acc': test_acc , 'epoch': epoch,}
        torch.save(state, model_path)
    print(f'Best Testing IoU Based On the Training:{best_test_iou}')
    print(f'Best Testing Accuracy Based On the Training:{Best_test_acc_BOT}')
    print(f'Time Elapsed:{end-start}\n')    
    with open(accuracy_file_path, 'a') as f:
        f.write(f'Best Testing IoU Based On the Training:{best_test_iou}\n')
        f.write(f'Best Testing Accuracy Based On the Training:{Best_test_acc_BOT}\n')
        f.write(f'Time Elapsed:{end-start}\n')


            
def check_class_performance(train_ce_loss, test_acc, curr_test_iou,net, epoch, model_path, accuracy_file_path, start, end):    
    global min_loss, Best_test_acc_BOT, best_test_iou
    if train_ce_loss<min_loss:
        print('Saving..')
        min_loss=train_ce_loss
        Best_test_acc_BOT=test_acc
        best_test_iou=curr_test_iou
        if isinstance(net, torch.nn.DataParallel):
            net_nwrap = net.module
        else:
            net_nwrap=net
        state = {'net': net_nwrap.state_dict(), 'test_iou': curr_test_iou, 'Test_acc': test_acc, 'epoch': epoch,}
        torch.save(state, model_path)
    
    print(f'Best Testing IoU Based On the Training:{best_test_iou}')
    print(f'Best Testing Accuracy Based On the Training:{Best_test_acc_BOT}')
    print(f'Time Elapsed:{end-start}\n')    
    with open(accuracy_file_path, 'a') as f:
        f.write(f'Best Testing IoU Based On the Training:{best_test_iou}\n')
        f.write(f'Best Testing Accuracy Based On the Training:{Best_test_acc_BOT}\n')
        f.write(f'Time Elapsed:{end-start}\n')
        
    

    
def check_performance(iou, curr_test_iou, train_ce_loss, test_acc, net, epoch, model_path, accuracy_file_path, start, end):
    global train_best_iou, best_test_iou, Best_test_acc_BOT, min_loss
    updated = False
    # Check for segment performance improvement
    if iou > train_best_iou:
        print('Saving based on segment performance improvement..')
        train_best_iou = iou
        best_test_iou = curr_test_iou
        Best_test_acc_BOT = test_acc
        updated = True

    # Check for class performance improvement
    if train_ce_loss < min_loss:
        print('Saving based on class performance improvement..')
        min_loss = train_ce_loss
        Best_test_acc_BOT = test_acc
        best_test_iou = curr_test_iou
        updated = True

    # Save the model if there was an update
    if updated:
        if isinstance(net, torch.nn.DataParallel):
            net_nwrap = net.module
        else:
            net_nwrap = net
        state = {'net': net_nwrap.state_dict(),'test_iou': curr_test_iou,'test_acc': test_acc,'epoch': epoch,}
        torch.save(state, model_path)
        
    print(f'Best Testing IoU Based On the Training:{best_test_iou}')
    print(f'Best Testing Accuracy Based On the Training:{Best_test_acc_BOT}')
    print(f'Time Elapsed:{end-start}\n')    
    with open(accuracy_file_path, 'a') as f:
        f.write(f'Best Testing IoU Based On the Training:{best_test_iou}\n')
        f.write(f'Best Testing Accuracy Based On the Training:{Best_test_acc_BOT}\n')
        f.write(f'Time Elapsed:{end-start}\n')


# In[ ]:


best_iou=0
best_acc=0

train_best_iou=0
best_test_iou=0

min_loss=1e10
test_acc=0
Best_test_acc_BOT=0

for epoch in range(start_epoch, args.max_epoch):
    start = time.time()
    iou,train_acc,train_ce_loss= train_epoch_Seg(epoch)
    curr_test_iou,test_acc,test_ce_loss=test_epoch_Seg(epoch)
    scheduler.step()
    end = time.time()
    
    if seg_ild and not (cls_ild):
        check_seg_performance(iou,curr_test_iou,test_acc,net,epoch,end,start,model_path,accuracy_file_path) 
    elif not(seg_ild) and cls_ild:
        check_class_performance(train_ce_loss, test_acc, curr_test_iou,net, epoch, model_path, accuracy_file_path, start, end)
    else:
        check_performance(iou, curr_test_iou, train_ce_loss, test_acc, net, epoch, model_path, accuracy_file_path, start, end)
        
        


# In[ ]:


all_results_file =os.path.join(current_working_directory,train_type,args.backbone_class,'all_results_file.txt')



from filelock import Timeout, FileLock
lock = FileLock(all_results_file[:-4]+'.lock', timeout=120)  # Timeout after two minutes
try:
    with lock:
        with open(all_results_file, 'a') as file:
            file.write(f'\n*************************************************************\n')
            file.write(f"**{name} 's Testing Accuracies**\n")
            file.write(f'***Total parameters in the model: {total_params/1e+6}***\n')            
            file.write(f'*****{formatted_datetime} Time*****\n')
            file.write(f'Best Testing IoU Based On the Training:{best_test_iou}\n')
            file.write(f'Best Testing Accuracy Based On the Training:{Best_test_acc_BOT}\n')
            file.write(f'*************************************************************\n\n')
except Timeout:
    print("Could not acquire the lock within 120 seconds.")


# In[ ]:


'''
# Add arguments
parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--max_epoch', default=150, type=int, help='resume from checkpoint')
parser.add_argument('--backbone_class', type=str, default='densenet161', choices=['densenet161', 'vgg19', 'resnet50', 'resnet34', 'mobilenet_v2', 'inception_v3'], help='resume from checkpoint')
parser.add_argument('--dataset', type=str, default='soybean_2_1', choices=['soybean_1_1', 'soybean_2_1', 'btf', 'hainan_leaf'], help='resume from checkpoint')
parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('--Seg_Size', default=448, type=int, help='Segmentation Dimension')
parser.add_argument('--num_classes', default=200, type=int, help='Number of Classes')



parser.add_argument('--DataParallel', action='store_true', help='Enable Data Parallel')
parser.add_argument('--seg_included', action='store_true', help='Enable Segmentation Training')
parser.add_argument('--cls_included', action='store_true', help='Enable Classification Training')
parser.add_argument('--freeze_all', action='store_true', help='Freeze the Encoder Module')
parser.add_argument('--MMANet', action='store_true', help='Using the MMANet')
parser.add_argument('--MGANet', action='store_true', help='Using the MGANet')
parser.add_argument('--maskguided', action='store_true', help='Using the MGANet')
parser.add_argument('--Unet', action='store_true', help='Using the Unet')
parser.add_argument('--mp', action='store_true', help='using automatic mixed precision training”')

parser.add_argument('--model_path', type=str , help='Use Pretrained Model')
'''


# In[ ]:


'''
Epoch: 0
Acc: 0.375% (3/800)| CE: 5.5615| Total Loss: 5.5615| IoU :0.0000
Acc: 0.000% (0/400)| CE: 6.0858|  Total Loss: 6.0858| IoU :0.0000
cur_iou:0.0,best_iou:0:
curr_Acc:0.0,best_Acc:0:
Saving..
Best Testing IoU Based On the Training:0.0
Best Testing Accuracy Based On the Training:0.0
Time Elapsed:20.380533456802368


Epoch: 1
Acc: 0.875% (7/800)| CE: 5.1327| Total Loss: 5.1327| IoU :0.0000
Acc: 3.500% (14/400)| CE: 4.9119|  Total Loss: 4.9119| IoU :0.0000
cur_iou:0.0,best_iou:0:
curr_Acc:3.5,best_Acc:3.5:
Saving..
Best Testing IoU Based On the Training:0.0
Best Testing Accuracy Based On the 

'''


# In[ ]:


# from copy import deepcopy
# net_original = deepcopy(net)


# x = torch.randn(8, 1, 28, 28)

# # Define a 2x2 max pooling layer
# pool = nn.MaxPool2d(kernel_size=2, stride=2)

# # Apply the pooling layer to the tensor
# y = pool(x)

# print(y.size())

# def compare_model_params(model1, model2):
#     # Get state dictionaries
#     state_dict1 = model1.state_dict()
#     state_dict2 = model2.state_dict()
    
#     total_unchanged=0
#     total_changed=0

#     # Compare sizes and if sizes match, compare values
#     for ((key1, param1), (key2, param2)) in zip(state_dict1.items(), state_dict2.items()):
        
#         if param1.size() != param2.size():
#             print(f"Mismatch in size for layer {key1}: {param1.size()} vs {param2.size()}")
#             continue  # Skip further comparison if sizes differ
#         # Check if the parameters are the same
#         if torch.equal(param1, param2):
#             total_unchanged+=1

#         else:
#             total_changed+=1
#             print(f"Parameters of layer {key1} differ.")
#     print('Changed, unchanged, Both',total_changed,total_unchanged,total_changed+total_unchanged)

# compare_model_params(net,net_original)

# import matplotlib.pyplot as plt
# plt.plot(train_iou)
# plt.show()
# plt.plot(test_iou)
# plt.show()


# print(test_iou[-1])
# print(train_iou[-1])
# test_iou[250]
# print(optimizer.param_groups[0]['lr'])


# In[ ]:


# def _check_layers(model, upto=None):
#     cnt, th = 0, 0
#     print('freeze layers:')
#     if upto:
#         th = upto
#     else:
#         th=float('inf')
#     for name, child in model.named_children():
#         cnt += 1
#         if cnt < th:
#             layer_type = type(child).__name__
#             for name2, params in child.named_parameters():
#                 #layer_type2 = type(params).__name__
                
#                 print(name, layer_type,name2, cnt,params.requires_grad) 
# #                 if params.requires_grad==True:
# #                     number_of_unfrozen_layers+=1



# # def _check_layers(model, prefix='', cnt=0, th=float('inf')):
# #     if cnt >= th:
# #         return
# #     for name, child in model.named_children():
# #         cnt += 1
# #         new_prefix = f"{prefix}{name}." if prefix else name
# #         layer_type = type(child).__name__
# #         if len(list(child.children())) > 0:  # Check if child has further sub-layers
# #             _check_layers(child, new_prefix, cnt, th)
# #         for name2, params in child.named_parameters(recurse=False):
# #             print(f"{new_prefix} ({layer_type}) {name2} {cnt} {params.requires_grad}")



# number_of_unfrozen_layers=0

# print('*******Features****************')
  
        
# _check_layers(model)


# # print('*******Classifier****************')

# # nnn=model.classifier
# # _check_layers(nnn)



# # print('*******Classifier****************')
# # for name, param in nnn.named_parameters():
# #     print(name,param.requires_grad)
# #     if param.requires_grad==True:
# #         number_of_unfrozen_layers+=1
    
    
# # print('**********Attention*************')

    
# # nnn=model.attention
# # for name, param in nnn.named_parameters():
# #     print(name,param.requires_grad)
# #     if param.requires_grad==True:
# #         number_of_unfrozen_layers+=1
    
    
    
# # print('**********Encoders1*************')

# # for i in range (1,6):
# #     nnn=model.Encoders[i]
# #     _check_layers(nnn)

# # print(f'Total unfrozzen:{number_of_unfrozen_layers} ')


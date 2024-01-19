#!/usr/bin/env python
# coding: utf-8

# In[1]:


print('**Code Starting Optunaaaaaaaaaaa**')

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

import optuna
from optuna.trial import TrialState

print('**End of Importing**')


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


if args.local_train==1:
    #current_working_directory = os.getcwd()    
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
        train_type=os.path.join(args.dataset+'-results-seg','Unet')
    else:
        train_type=os.path.join(args.dataset+'-results-seg','3PlusUnet')

else:
    if args.unet:
        train_type=os.path.join(args.dataset+'-results-seg-cls','Unet')
    else:
        train_type=os.path.join(args.dataset+'-results-seg-cls','3PlusUnet')

folder_path =os.path.join(current_working_directory,train_type,args.backbone_class,name,formatted_datetime)
accuracy_file_path =os.path.join(folder_path,'Accuracies_Iou.txt')
    
    
print('File path:',accuracy_file_path)
    
    
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    

models_folder=current_working_directory+'/checkpoint'
if not os.path.exists(models_folder):
    os.makedirs(models_folder)


# In[ ]:


train = Leafvein(args,crop=[448,448],hflip=True,vflip=False,erase=False,mode='train')
test = Leafvein(args,crop=[448,448],mode='test')
model_path= f'{current_working_directory}/checkpoint/{name}_{formatted_datetime}.pth'


# In[ ]:


class_loss_fn = nn.CrossEntropyLoss()
seg_loss_fn   = nn.BCEWithLogitsLoss()


# In[ ]:


best_iou=0
best_acc=0

train_best_iou=0
best_test_iou=0

min_loss=1e10
test_acc=0
Best_test_acc_BOT=0

def objective(trial):
    start = time.time()
    # Generate the model.
    model=MMANET(trial,num_classes=num_classes,MANet=MANet,MMANet=MMANet,mask_guided=mask_guided,seg_included=seg_ild,freeze_all=freeze_all,Unet=args.unet,deform_expan=args.deform_expan)
    model=model.to(device)
    model = torch.nn.DataParallel(model)
    
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch)
    # Get the dataset.
    train_loader = DataLoader(train, batch_size=batchsize, shuffle=True)
    valid_loader = DataLoader(test, batch_size=batchsize, shuffle=False)
    
    

    # Training of the model.
    for epoch in range(EPOCHS):
        start = time.time()        
        iou,train_acc,train_ce_loss= train_epoch_Seg(epoch)
        curr_test_iou,test_acc,test_ce_loss=test_epoch_Seg(epoch)
        scheduler.step()
        end = time.time()

        print(f'Epoch:{epoch}, Elapsed Time:{end-start}')

        trial.report(curr_test_iou, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return curr_test_iou


# In[ ]:


input_size = [args.seg_size, args.seg_size]
new_size = [size // 2 for size in input_size]

def train_epoch_Seg(epoch):
    print('\nEpoch: %d' % epoch)
    
    global new_size
   
    
    model.train()
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
             

    return averageIoU,accuracy,test_ce_loss


# In[ ]:


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100, timeout=600)

pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])


# In[ ]:


print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))
print("  Number of pruned trials: ", len(pruned_trials))
print("  Number of complete trials: ", len(complete_trials))

print("Best trial:")
trial = study.best_trial

print("  Value: ", trial.value)

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))


# In[ ]:


# best_iou=0
# best_acc=0

# train_best_iou=0
# best_test_iou=0

# min_loss=1e10
# test_acc=0
# Best_test_acc_BOT=0

# for epoch in range(start_epoch, args.max_epoch):
#     start = time.time()
#     iou,train_acc,train_ce_loss= train_epoch_Seg(epoch)
#     curr_test_iou,test_acc,test_ce_loss=test_epoch_Seg(epoch)
#     scheduler.step()
#     end = time.time()
    
#     if seg_ild and not (cls_ild):
#         check_seg_performance(iou,curr_test_iou,test_acc,net,epoch,end,start,model_path,accuracy_file_path) 
#     elif not(seg_ild) and cls_ild:
#         check_class_performance(train_ce_loss, test_acc, curr_test_iou,net, epoch, model_path, accuracy_file_path, start, end)
#     else:
#         check_performance(iou, curr_test_iou, train_ce_loss, test_acc, net, epoch, model_path, accuracy_file_path, start, end)
        
        


# In[ ]:


# def check_seg_performance(iou,curr_test_iou,test_acc,net,epoch,end,start,model_path,accuracy_file_path):
#     global train_best_iou,best_test_iou,Best_test_acc_BOT
#     if iou>train_best_iou:
#         print('Saving..')
#         train_best_iou=iou
#         best_test_iou=curr_test_iou
#         Best_test_acc_BOT=test_acc
#         if isinstance(net, torch.nn.DataParallel):
#             net_wrap = net.module
#         else:
#             net_wrap=net
#         state = {'net': net_wrap.state_dict(), 'test_iou': best_test_iou, 'Test_acc': test_acc , 'epoch': epoch,}
#         torch.save(state, model_path)
#     print(f'Best Testing IoU Based On the Training:{best_test_iou}')
#     print(f'Best Testing Accuracy Based On the Training:{Best_test_acc_BOT}')
#     print(f'Time Elapsed:{end-start}\n')    
#     with open(accuracy_file_path, 'a') as f:
#         f.write(f'Best Testing IoU Based On the Training:{best_test_iou}\n')
#         f.write(f'Best Testing Accuracy Based On the Training:{Best_test_acc_BOT}\n')
#         f.write(f'Time Elapsed:{end-start}\n')
        
#     wandb.log({"Best Testing IoU BOT": best_test_iou,
#                "Best Testing Accuracy BOT":Best_test_acc_BOT,
#                "Elapsed Time":end-start},step=epoch)


            
# def check_class_performance(train_ce_loss, test_acc, curr_test_iou,net, epoch, model_path, accuracy_file_path, start, end):    
#     global min_loss, Best_test_acc_BOT, best_test_iou
#     if train_ce_loss<min_loss:
#         print('Saving..')
#         min_loss=train_ce_loss
#         Best_test_acc_BOT=test_acc
#         best_test_iou=curr_test_iou
#         if isinstance(net, torch.nn.DataParallel):
#             net_nwrap = net.module
#         else:
#             net_nwrap=net
#         state = {'net': net_nwrap.state_dict(), 'test_iou': curr_test_iou, 'Test_acc': test_acc, 'epoch': epoch,}
#         torch.save(state, model_path)
    
#     print(f'Best Testing IoU Based On the Training:{best_test_iou}')
#     print(f'Best Testing Accuracy Based On the Training:{Best_test_acc_BOT}')
#     print(f'Time Elapsed:{end-start}\n')    
#     with open(accuracy_file_path, 'a') as f:
#         f.write(f'Best Testing IoU Based On the Training:{best_test_iou}\n')
#         f.write(f'Best Testing Accuracy Based On the Training:{Best_test_acc_BOT}\n')
#         f.write(f'Time Elapsed:{end-start}\n')
        
    

    
# def check_performance(iou, curr_test_iou, train_ce_loss, test_acc, net, epoch, model_path, accuracy_file_path, start, end):
#     global train_best_iou, best_test_iou, Best_test_acc_BOT, min_loss
#     updated = False
#     # Check for segment performance improvement
#     if iou > train_best_iou:
#         print('Saving based on segment performance improvement..')
#         train_best_iou = iou
#         best_test_iou = curr_test_iou
#         Best_test_acc_BOT = test_acc
#         updated = True

#     # Check for class performance improvement
#     if train_ce_loss < min_loss:
#         print('Saving based on class performance improvement..')
#         min_loss = train_ce_loss
#         Best_test_acc_BOT = test_acc
#         best_test_iou = curr_test_iou
#         updated = True

#     # Save the model if there was an update
#     if updated:
#         if isinstance(net, torch.nn.DataParallel):
#             net_nwrap = net.module
#         else:
#             net_nwrap = net
#         state = {'net': net_nwrap.state_dict(),'test_iou': curr_test_iou,'test_acc': test_acc,'epoch': epoch,}
#         torch.save(state, model_path)
        
#     print(f'Best Testing IoU Based On the Training:{best_test_iou}')
#     print(f'Best Testing Accuracy Based On the Training:{Best_test_acc_BOT}')
#     print(f'Time Elapsed:{end-start}\n')    
#     with open(accuracy_file_path, 'a') as f:
#         f.write(f'Best Testing IoU Based On the Training:{best_test_iou}\n')
#         f.write(f'Best Testing Accuracy Based On the Training:{Best_test_acc_BOT}\n')
#         f.write(f'Time Elapsed:{end-start}\n')


# In[ ]:


# run.finish()

# all_results_file =os.path.join(current_working_directory,train_type,args.backbone_class,'all_results_file.txt')



# from filelock import Timeout, FileLock
# lock = FileLock(all_results_file[:-4]+'.lock', timeout=120)  # Timeout after two minutes
# try:
#     with lock:
#         with open(all_results_file, 'a') as file:
#             file.write(f'\n*************************************************************\n')
#             file.write(f"**{name} 's Testing Accuracies**\n")
#             file.write(f'***Total parameters in the model: {total_params/1e+6}***\n')            
#             file.write(f'*****{formatted_datetime} Time*****\n')
#             file.write('********Training IOU\n')
#             file.write(f'Training IOU:{iou}\n')
#             file.write(f'Best Training IOU:{train_best_iou}\n')
#             file.write('********Testing IOU\n')
#             file.write(f'The overall Very Best Testing IOU :{best_iou}\n')
#             file.write(f'Best Testing IoU Based On the Training:{best_test_iou}\n')
#             file.write('********\n')
#             file.write('Training Acc*****\n')
#             file.write(f'Training Loss:{train_ce_loss}\n')
#             file.write(f'Best Testing Accuracy Based On the Training:{Best_test_acc_BOT}\n')
#             file.write(f'The overall Very Best Testing Accuracy :{best_acc}\n')
#             file.write(f'*************************************************************\n\n')
# except Timeout:
#     print("Could not acquire the lock within 120 seconds.")


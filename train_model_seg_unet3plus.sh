#!/bin/bash

# Check if an argument is provided
if [ $# -eq 0 ]; then
    echo "No arguments provided. Please provide an argument (1-10)."
    exit 1
fi


# Get the argument
ARG=$1
model=$2
local_train=$3

echo "Model Number: $ARG"
echo "Backbone: $model"



if [ "$local_train" = true ]; then

    DATA_DIR="--data_dir /home/pupil/rmf3mc/Documents/ModelProposing/MGANet/FinalTouches_AMP/data"
    Base_model="--model_path /home/pupil/rmf3mc/Documents/ModelProposing/MGANet/FinalTouches_AMP/checkpoint/file.pth"
    Training_MC="--local_train 1"
else

    DATA_DIR="--data_dir /work/MGANet/data"
    Base_model="--model_path  /mnt/mywork/all_backbones/checkpoint/densenet161_MGANet_Classification_03:55-21-12-2023.pth"
    Training_MC="--local_train 0"
fi

echo "Where training: $local_train"


# Define the base command
BASE_CMD="python -u Train.py"


# Run the command based on the argument
case $ARG in

 1)
    $BASE_CMD --mmanet            --seg_ild --freeze_all --dataparallel $DATA_DIR      --backbone_class  $model  $Base_model  --att_from 1  $Training_MC
    ;;

3)
    $BASE_CMD --mmanet              --seg_ild --freeze_all --dataparallel $DATA_DIR      --backbone_class  $model  $Base_model   --att_from 2  $Training_MC
    ;;

5)
    $BASE_CMD --mmanet              --seg_ild --freeze_all --dataparallel $DATA_DIR      --backbone_class  $model  $Base_model   --att_from 3  $Training_MC
    ;;

7)
    $BASE_CMD --mmanet              --seg_ild --freeze_all --dataparallel $DATA_DIR      --backbone_class  $model  $Base_model   --att_from 4  $Training_MC
    ;;

9)
    $BASE_CMD --mmanet              --seg_ild --freeze_all --dataparallel $DATA_DIR      --backbone_class  $model  $Base_model   --att_from 5  $Training_MC
    ;;

11)
    $BASE_CMD --mmanet              --seg_ild --freeze_all --dataparallel $DATA_DIR      --backbone_class  $model  $Base_model   --att_from 6  $Training_MC
    ;;

# The following commands are commented out as they contain the --fsds argument
# 2)
#    $BASE_CMD --mmanet  --fsds      --seg_ild --freeze_all --dataparallel $DATA_DIR      --backbone_class  $model  $Base_model   --att_from 1  $Training_MC
#    ;;

# 4)
#    $BASE_CMD --mmanet  --fsds      --seg_ild --freeze_all --dataparallel $DATA_DIR      --backbone_class  $model  $Base_model   --att_from 2  $Training_MC
#    ;;

# 6)
#    $BASE_CMD --mmanet  --fsds      --seg_ild --freeze_all --dataparallel $DATA_DIR      --backbone_class  $model  $Base_model   --att_from 3  $Training_MC
#    ;;

# 8)
#    $BASE_CMD --mmanet  --fsds      --seg_ild --freeze_all --dataparallel $DATA_DIR      --backbone_class  $model  $Base_model   --att_from 4  $Training_MC
#    ;;

# 10)
#    $BASE_CMD --mmanet  --fsds      --seg_ild --freeze_all --dataparallel $DATA_DIR      --backbone_class  $model  $Base_model   --att_from 5  $Training_MC
#    ;;

# 12)
#    $BASE_CMD --mmanet  --fsds      --seg_ild --freeze_all --dataparallel $DATA_DIR      --backbone_class  $model  $Base_model   --att_from 6  $Training_MC
#    ;;


    *)
        echo "Invalid argument. Please provide a number between 1 and 10."
        exit 1
        ;;
esac

echo "Command executed."

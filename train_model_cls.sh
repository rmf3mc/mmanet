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



# Define the base command
BASE_CMD="python -u Train.py"



if [ "$local_train" = true ]; then

    DATA_DIR="--data_dir ./data"

else

    DATA_DIR="--data_dir /mnt/mywork/data"

fi

echo "Where training: $local_train"






# Run the command based on the argument
case $ARG in
    1)
        $BASE_CMD --mmanet              --cls_ild --dataparallel  $DATA_DIR  --backbone_class  $model 
        ;;

    2)
        $BASE_CMD --mmanet --maskguided --cls_ild --dataparallel $DATA_DIR   --backbone_class  $model
        ;;

    3)
        $BASE_CMD                       --cls_ild --dataparallel $DATA_DIR   --backbone_class  $model
        ;;


    4)
        $BASE_CMD --mmanet              --cls_ild --dataparallel $DATA_DIR   --backbone_class  $model     --amp
        ;;

    5)
        $BASE_CMD --mmanet --maskguided --cls_ild --dataparallel $DATA_DIR   --backbone_class  $model     --amp   
        ;;


    6)
        $BASE_CMD                       --cls_ild --dataparallel $DATA_DIR   --backbone_class  $model     --amp
        ;;



    *)
        echo "Invalid argument. Please provide a number between 1 and 10."
        exit 1
        ;;
esac

echo "Command executed."

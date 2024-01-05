

pvc_dep="dep3-64bf89cc4c-7v5wn:/mnt/mywork/all_backbones/"

kubectl cp Train.py     $pvc_dep
kubectl cp Train.ipynb     $pvc_dep
kubectl cp backboneModels.py     $pvc_dep
kubectl cp unet3.py     $pvc_dep
kubectl cp utils.py     $pvc_dep 
kubectl cp leafvein2.py     $pvc_dep 
kubectl cp train_model_seg.sh     $pvc_dep 
kubectl cp train_model_cls.sh     $pvc_dep 



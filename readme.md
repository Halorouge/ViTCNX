Steps

1. Put the image of dataset to "./dataset/dataset0..1..2..3" ,and named the folds "covid","normal","others"
2. Put the initial weight in "./init"
3. In main_train.py set the numble of dataset and classes
4. Run main_train.py 
5. In esn.py also set the numble of dataset and classes
6. The results of single model is in  "./res_dir/res.txt"
7. Run esn.py get esn results

Downloads:

dataset1:https://www.kaggle.com/datasets/plameneduardo/a-covid-multiclass-dataset-of-ct-scans

dataset2:https://www.kaggle.com/datasets/plameneduardo/sarscov2-ctscan-dataset

dataset3:https://github.com/UCSD-AI4H/COVID-CT

dataset0:all of dataset1 to dataset3

convnext_tiny_1k_224_ema.pth:https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth

vit_base_patch16_224_in21k.pth:https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth

Requirements:

python 3.8

pytorch 1.10.2


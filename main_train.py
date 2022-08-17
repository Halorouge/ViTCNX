import pandas as pd
from vit_model import train as vit_train
from vit_model import predict as vit_predict
from convnext_model import train as convnext_train
from convnext_model import predict as convnext_predict
from utils import read_split_data,mk_dir


TRAIN = True
PREDICT = True
PRE_train = True
n_class = 3 #numble of classes
ep = 2 
t_size = 0.2 #ratio of testset
seed = 8 #random seed of dataset split
dataset = 0 
# 0:fulldataset with 7398 images 
# 1:dataset1 with 4171 images 
# 2:dataset2 with 2481 images
# 3:dataset3 with 746  images

if __name__ == '__main__':
    data_path = './dataset/dataset' + str(dataset)
    mk_dir(dataset)

    train, train_label, test, test_label = read_split_data(data_path, t_size ,n_class=n_class, seed = seed)
    if TRAIN:
        vit_train(train, train_label, classes=n_class, device='cuda:0', val=False, data=dataset, epochs=ep, init=PRE_train)
        convnext_train(train, train_label ,classes=n_class, device='cuda:0', val=False, data=dataset, epochs=ep, init=PRE_train)
        print('train over')
    if PREDICT:
        score1 = vit_predict(test, test_label, num_class=n_class, data=dataset)
        score2 = convnext_predict(test, test_label, num_class=n_class, data=dataset)
        pd.DataFrame(test_label).to_csv('./res_dir/label.csv', header=None, index=None)
        print('predict over')

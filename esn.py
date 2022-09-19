from matplotlib.ft2font import HORIZONTAL
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve, precision_recall_curve, accuracy_score, recall_score, f1_score, precision_score
import pylab as plt
import warnings;warnings.filterwarnings('ignore')

#   covid 0  normal 1 cp  2

path = 'res_dir'
num = 3  #numble of classes
dataset = 0

vit = pd.read_csv(path + '/label/dataset'+ str(dataset) + '/vit.csv', header=None, index_col=None).to_numpy()
convnext = pd.read_csv(path + '/label/dataset'+ str(dataset) + '/convnext.csv', header=None, index_col=None).to_numpy()
label = pd.read_csv(path + '/label.csv', header=None, index_col=None).to_numpy()

namelists = ['ConvNeXt','Vision Transformer','ViTCNX']
colors = ['#A9C7C5','#95BCCC','#F7AD19']

def caltp(pred,label): #calculate tp fp tn fn
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    numble = len(label)
    for i in range(numble):
        if pred[i] == 1 and label[i] == 1:
            tp += 1
        elif pred[i] == 1 and label[i] == 0:
            fp += 1
        elif pred[i] == 0 and label[i] == 0:
            tn += 1
        elif pred[i] == 0 and label[i] == 1:
            fn += 1
    if tp + fp + tn + fn == numble :
        print("TP={},TN={},FP={},FN={}".format(tp,tn,fp,fn))
        return tp,tn,fp,fn
    else:
        print("not equal")
        return -1,-1,-1,-1

def drawmatrix(tp,tn,fp,fn,name):  #draw fusion matrix
    classes = ['Covid','Healthy']
    confusion_matrix = np.array(([tp,fp],[fn,tn]))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)  
    plt.title('{}'.format(name))
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    thresh = confusion_matrix.max() / 2.
    iters = np.reshape([[[i,j] for j in range(len(classes))] for i in range(len(classes))],(confusion_matrix.size,2))
    for i, j in iters:
        plt.text(j, i, format(confusion_matrix[i, j]),fontsize=30,horizontalalignment = 'center',verticalalignment = 'center')  
    plt.ylabel('Real label')
    plt.xlabel('Prediction')
    plt.tight_layout()
    plt.show()

def draw_auc(score,names,colors):  #draw auc
    line,colunm = np.shape(score)
    plt.figure(figsize=(10,10))
    # plt.title('Validation ROC')
    for i in range(colunm):
        fpr,tpr,th= roc_curve(label,score[:,i])
        re_auc = auc(fpr,tpr)
        plt.plot(fpr,tpr, colors[i], label = '{} (AUC = {:.4f})'.format(names[i],re_auc))

    plt.legend(loc = 'lower right', prop={'size': 16})
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.show()

def draw_aupr(score,names,colors):  #draw aupr
    line,colunm = np.shape(score)
    plt.figure(figsize=(10,10))
    # plt.title('Validation ROC')
    for i in range(colunm):
        pre,rec,thres= precision_recall_curve(label,score[:,i])
        re_aupr = auc(rec,pre)
        plt.plot(rec,pre, colors[i], label = '{} (AUC = {:.4f})'.format(names[i],re_aupr))

    plt.legend(loc = 'lower left', prop={'size': 16})
    plt.xlim([-0.1, 1.1])
    plt.ylim([0.3, 1.1])
    plt.ylabel('Precision', fontsize=12)
    plt.xlabel('Recall', fontsize=12)
    plt.show()

def drawlinechart(x_data,y_data,colors,name):  #draw linechart of metric:x_data are metrics and y_data are scores of metrics
    shape = ['s','o','^']
    line_count , x_count = y_data.shape
    for i in range(line_count):
        plt.plot(x_data,y_data[i],colors[i]+'-'+shape[i],alpha = 0.5,linewidth = 1,label = name[i])
    
    plt.legend()
    plt.xlabel('Metrics')
    plt.ylabel('Scores')
    plt.show()

def drawhistogram(x_data,y_data,colors,name):  #draw histogram of metric:x_data are metrics and y_data are scores of metrics
    bar_width = 0.25
    tick_label = x_data
    line_count , x_count = y_data.shape
    x = np.arange(x_count)
    x = x*2
    for i in range(line_count):
        plt.bar(x + i * bar_width , y_data[i],bar_width,align = 'center', ec='k',color = colors[i] ,  label = name[i])
    plt.xlabel('Metrics')
    plt.ylabel('Scores')
    plt.xticks(x+bar_width*(line_count/2), tick_label)
    plt.legend(bbox_to_anchor=(1.05, 0), loc= 3 , borderaxespad = 0)
    plt.gcf().subplots_adjust(left=None,top=None,bottom=None, right=0.7)
    plt.show()


def ensemble(num,vit,convnext,label,namelists,colors):  #ensemble
    if num == 2:
        score = np.ones(label.shape)
        score = np.hstack([score, convnext])
        score = np.hstack([score, vit])
        score = score[:, 1:]
        score = 1 - score
        label = 1 - label        

        # # draw fusion matrix
        # for i in range(np.shape(score)[1]):
        #     print("*********{}**********".format(namelists[i]))
        #     tp,tn,fp,fn = caltp(allpred[:,i],label)
        #     if tp != -1:
        #         drawmatrix(tp,tn,fp,fn,namelists[i])
        #     print("")

        pa = 0.6
        score_res = pa*score[:,1]+(1-pa)*score[:,0]
        score_res_up = score_res[:,np.newaxis]
        score = np.hstack([score,score_res_up])
        vcfpr, vctpr, vcthreshold = roc_curve(label, score_res)
        vcpre, vcrec_, _ = precision_recall_curve(label, score_res)
        pred = np.zeros(len(label))
        pred[score_res >= 0.5] = 1
        acc = accuracy_score(label, pred)
        rec = recall_score(label, pred)
        f1 = f1_score(label, pred)
        Pre = precision_score(label, pred)
        au = auc(vcfpr, vctpr)
        apr = auc(vcrec_, vcpre)

        # draw_auc(score,namelists,colors)  
        # draw_aupr(score,namelists,colors)

        print('{}-Class expriments,Precision is :{}'.format(num,Pre))
        print('{}-Class expriments,Recall is :{}'.format(num,rec))
        print('{}-Class expriments,ACC is: {}'.format(num,acc))
        print('{}-Class expriments,F1 is: {}'.format(num,f1))
        print('{}-Class expriments,AUC is: {}'.format(num,au))
        print('{}-Class expriments,AUPR is :{}'.format(num,apr))

    elif num == 3:
        score = np.zeros(vit.shape)
        score += 0.6*vit
        score += 0.4*convnext
        esn_pred = np.argmax(score, axis=1)
        vit_pred = np.argmax(vit,axis = 1)
        convnext_pred = np.argmax(convnext,axis = 1)
        preds = np.stack((convnext_pred,vit_pred,esn_pred),axis = 0)
        for i in range(len(namelists)):
            print("")
            print('******************{}********************'.format(namelists[i]))
            pred = preds[i]
            rec = recall_score(label, pred, average='macro')
            f1 = f1_score(label, pred, average='macro')
            Pre = precision_score(label, pred, average='macro')
            acc = accuracy_score(label, pred)
            print('{}-Class expriments,Pre is :{}'.format(num,Pre))
            print('{}-Class expriments,Rec is :{}'.format(num,rec))
            print('{}-Class expriments,ACC is: {}'.format(num,acc))
            print('{}-Class expriments,F1  is: {}'.format(num,f1))

ensemble(num,vit,convnext,label,namelists,colors)


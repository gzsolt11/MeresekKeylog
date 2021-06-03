from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cityblock
from sklearn.metrics import roc_curve, auc
from sklearn.svm import OneClassSVM
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import svm
import pandas as pd
import numpy as np
import pathlib
import os


def beolvas(filename):
    fn = pathlib.Path(__file__).parent / 'processed_data'/pathlib.Path(filename)
    df_DSL = pd.read_csv(fn)
    return df_DSL


def evaluateEER(false_positive_rate, true_positive_rate):
    missrates = 1 - true_positive_rate # false negative rate
    
    #fnr = 1 - tpr
    EER_fpr = false_positive_rate[np.argmin(np.absolute((missrates - false_positive_rate)))]
    EER_fnr = missrates[np.argmin(np.absolute((missrates - false_positive_rate)))]
    EER = 0.5 * (EER_fpr + EER_fnr)

    return EER


def split_data(data,user,train_size, test_size):

    genuine_user_data = data.loc[data.user_id == user]

    test_genuine = genuine_user_data[train_size:]

    imposter_data = data.loc[data.user_id != user]
    imposter_data = imposter_data.drop(["user_id"],axis=1)
    test_imposter = imposter_data.sample(n=test_size, random_state=0)
    #test_imposter = imposter_data
                
    train_genuine = genuine_user_data[:train_size].drop(['user_id'],axis=1)

    test_genuine = genuine_user_data[train_size:].drop(['user_id'],axis=1)
    #test_imposter = imposter_data.groupby("user_id").drop(['user_id'],axis=1)
    #test_imposter = imposter_data.groupby("user_id").head(20).drop(['user_id'],axis=1)
    #test_imposter = shuffle(test_imposter,random_state = 0)
    #test_imposter = test_imposter.head(15)

    return [train_genuine, test_genuine, test_imposter]

def manhattan_scaled(train,test_genuine,test_imposter):
    mean_vector = train.mean().values
    mad_vector  = train.mad().values

    user_scores = []
    imposter_scores = []

    for i in range(test_genuine.shape[0]):
        cur_score = 0
        for j in range(len(mean_vector)):
            if(mad_vector[j] == 0.0):
                mad_vector[j] = 0.005
            cur_score = cur_score + abs(test_genuine.iloc[i].values[j] - mean_vector[j]) / mad_vector[j]

        cur_score = 1/(1+cur_score)
        user_scores.append(cur_score)
    
    for i in range(test_imposter.shape[0]):
        cur_score = 0
        for j in range(len(mean_vector)):
            cur_score = cur_score +  abs(test_imposter.iloc[i].values[j] - mean_vector[j]) / mad_vector[j]
        cur_score = 1/(1+cur_score)
        imposter_scores.append(cur_score)

    #print(user_scores)
    #print(imposter_scores)

    return [user_scores, imposter_scores]


def euclidean_distance(train,test_genuine,test_imposter,userId):
    mean_vector = train.mean().values
    
    plt.rcParams["figure.figsize"] = (10,5)
    font2 = {'family':'serif','size':15}

    user_scores = []
    imposter_scores = []

    for i in range(test_genuine.shape[0]):
        cur_score = 0
        cur_score = euclidean(test_genuine.iloc[i].values, mean_vector)
        cur_score = 1/(1+cur_score)
        user_scores.append(cur_score)
    
    for i in range(test_imposter.shape[0]):
        cur_score = 0
        cur_score = euclidean(test_imposter.iloc[i].values, mean_vector)
        cur_score = 1/(1+cur_score)
        imposter_scores.append(cur_score)

    return [user_scores, imposter_scores]

def svm_distance(train,test_genuine,test_imposter):
    model = OneClassSVM(gamma='scale')
    model.fit(train)
    user_scores = model.score_samples(test_genuine)
    imposter_scores = model.score_samples(test_imposter)
    user_scores = user_scores.tolist()
    imposter_scores = imposter_scores.tolist()

    return [user_scores, imposter_scores]

def plotEERandFeatures(thresholds,false_positive_rate, true_positive_rate, test_genuine_data, train_data, user_id,auc,eer):
    mean_vector = train_data.mean().values

    font2 = {'family':'serif','size':15}
    plt.figure(figsize=(15,7))
    plt.suptitle(str(user_id)+'. user', fontdict = font2)
    plt.subplot(2, 1, 1)
    plt.title("AUC:"+("%.2f" % auc) + " EER:"+("%.2f" % eer))
    plt.plot(thresholds, 1-true_positive_rate)
    plt.plot(thresholds, false_positive_rate)
    
    plt.subplot(2, 1, 2)
    # tesztelo adatok
    for i in range(0,test_genuine_data.shape[0]):
        plt.plot(test_genuine_data.iloc[i],color='brown', linestyle = '--', linewidth=0.8)

    #tanito adatok
    for i in range(0,train_data.shape[0]):
        plt.plot(train_data.iloc[i],color='#008000')
    
    # atlag abrazolasa
    plt.plot(mean_vector,color='#DC143C',linewidth=5)
    fn = pathlib.Path(__file__).parent / 'exportimages'

    if not fn.exists():
        os.mkdir(fn)

    user_string = str(user_id)+'.png'
    plt.savefig(fn / user_string)

    #plt.show()
    plt.clf()
    plt.cla()
    plt.close()


def manhattan_main(filename,pin_digit_number,train_size,test_size):
    data = beolvas(filename)
    users = data["user_id"]
    data = (data-data.min())/(data.max()-data.min())
    data.drop(["user_id"],axis=1)
    data["user_id"] = users
    users = data['user_id'].unique()

    error_auc_mean_dev = pd.DataFrame([],columns=['Classifier','EER(avg)','EER(std)','AUC(avg)','AUC(std)'])
    
    classifiers = ["Manhattan Scaled","Euclidean","SVM"]
    for i in range(0,len(classifiers)):
        result_data_frame = pd.DataFrame([],columns=['user_id','EER','AUC'])

        counter = 1
        for user in users:
            train, test_genuine, test_imposter = split_data(data,user,train_size,test_size)

            if i == 0:
                user_scores, imposter_scores = manhattan_scaled(train,test_genuine,test_imposter)
            elif i == 1:
                user_scores, imposter_scores = euclidean_distance(train,test_genuine,test_imposter,counter)
            elif i == 2:
                user_scores, imposter_scores = svm_distance(train, test_genuine, test_imposter)
            
            labels = [0]*len(user_scores) + [1]*len(imposter_scores)
            fpr, tpr, thresholds = roc_curve(labels, user_scores + imposter_scores, pos_label=0)
            roc_auc = auc(fpr, tpr)
            
            if i == 1:
                plotEERandFeatures(thresholds,fpr,tpr,test_genuine,train,counter,roc_auc,evaluateEER(fpr, tpr))
            
            counter += 1

            result_data_frame = result_data_frame.append({'user_id':user,'EER':evaluateEER(fpr, tpr), 'AUC':roc_auc},ignore_index=True)
            
           
        #print(result_data_frame.head(10))

        error_avg = np.average(result_data_frame['EER'])
        error_std = np.std(result_data_frame['EER'])
        AUC_avg = np.average(result_data_frame['AUC'])
        AUC_std = np.std(result_data_frame['AUC'])

        
        error_auc_mean_dev = error_auc_mean_dev.append({'Classifier':classifiers[i] , 'EER(avg)':error_avg,'EER(std)':error_std,'AUC(avg)':AUC_avg,'AUC(std)':AUC_std}, ignore_index=True)



    return error_auc_mean_dev
    

def main():
    error_auc_mean_dev = manhattan_main("out6digit.csv",6,10,10)
    print(error_auc_mean_dev)
    

main()



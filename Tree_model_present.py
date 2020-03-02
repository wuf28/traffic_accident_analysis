import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
import math
import matplotlib.pyplot as plt
import plotly as py
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
def model_Tree_result_present(modellr,train_x,train_y,test_x,test_y,max_dep,split_type,crt_type,classweight):
  modellr.fit(train_x, train_y)
  print("tree parameter",modellr.get_params())
  print("tree depth",modellr.get_depth())
  cm_test = confusion_matrix(test_y, modellr.predict(test_x))
  cm_train = confusion_matrix(train_y, modellr.predict(train_x))
  print ("Decision Tree Train Accuracy : ", modellr.score(train_x,train_y),'\n')
  print ("Decision Tree Test Accuracy : ", modellr.score(test_x,test_y),'\n')
  return cm_test,cm_train,modellr

from sklearn import tree
def test_parameter_tree(X_train, y_train, test_x, test_y, max_dep, split_type, crt_type, classweight):
  print('1：No Injury\n 2:Injury\n 3:Fatal\n')
  l = ['sev_T1_P1','sev_T2_P2','sev_T3_P3','sev_T1_P2','sev_T1_P3','sev_T2_P1','sev_T2_P3','sev_T3_P1','sev_T3_P2','Accuracy']
  result =pd.DataFrame(columns=l)
  all_model = []
  i =1
  # max_dep = 40
  # for split in ["best","random"]:
  for split in split_type:
    for crt in crt_type:
    # for crt in ["gini","entropy"]:
      train_overall=[]
      overall=[]
      non_injur=[]
      injur=[]
      fatal=[]
      train_non_injur=[]
      train_injur=[]
      train_fatal=[]
      for dep in range(max_dep):
        clf = tree.DecisionTreeClassifier(class_weight=classweight, criterion=crt, splitter=split, max_depth=dep+1)
        cm_test, cm_train, model = model_Tree_result_present(clf,X_train,y_train,test_x,test_y,max_dep,split_type,crt_type,classweight)
        rows1 = {'sev_T1_P1':cm_test[0,0]/np.sum(cm_test[0]),'sev_T2_P2':cm_test[1,1]/np.sum(cm_test[1]),'sev_T3_P3':cm_test[2,2]/np.sum(cm_test[2]),'sev_T1_P2':cm_test[0,1]/np.sum(cm_test[0]),'sev_T1_P3':cm_test[0,2]/np.sum(cm_test[0]),'sev_T2_P1':cm_test[1,0]/np.sum(cm_test[1]),'sev_T2_P3':cm_test[1,2]/np.sum(cm_test[1]),'sev_T3_P1':cm_test[2,0]/np.sum(cm_test[2]),'sev_T3_P2':cm_test[2,1]/np.sum(cm_test[2]),'Accuracy':np.trace(cm_test)/cm_test.sum()}
        result = result.append(pd.DataFrame(rows1, index = ['test {}'.format(i)]))
        rows2 = {'sev_T1_P1':cm_train[0,0]/np.sum(cm_train[0]),'sev_T2_P2':cm_train[1,1]/np.sum(cm_train[1]),'sev_T3_P3':cm_train[2,2]/np.sum(cm_train[2]),'sev_T1_P2':cm_train[0,1]/np.sum(cm_train[0]),'sev_T1_P3':cm_train[0,2]/np.sum(cm_train[0]),'sev_T2_P1':cm_train[1,0]/np.sum(cm_train[1]),'sev_T2_P3':cm_train[1,2]/np.sum(cm_train[1]),'sev_T3_P1':cm_train[2,0]/np.sum(cm_train[2]),'sev_T3_P2':cm_train[2,1]/np.sum(cm_train[2]),'Accuracy':np.trace(cm_train)/cm_train.sum()}
        result = result.append(pd.DataFrame(rows2, index = ['train {}'.format(i)]))
        all_model.append(model)
        i+=1
        overall.append(np.trace(cm_test)/cm_test.sum())
        train_overall.append(np.trace(cm_train)/cm_train.sum())
        non_injur.append(cm_test[0,0]/np.sum(cm_test[0]))
        train_non_injur.append(cm_train[0,0]/np.sum(cm_train[0]))
        injur.append(cm_test[1,1]/np.sum(cm_test[1]))
        train_injur.append(cm_train[0,0]/np.sum(cm_train[0]))
        fatal.append(cm_test[2,2]/np.sum(cm_test[2]))
        train_fatal.append(cm_train[0,0]/np.sum(cm_train[0]))

      plt.figure(figsize=(14,10),dpi=60)
      x_axis = range(1,max_dep+1)
      line1, =plt.plot(x_axis,overall, label="test_overall",linestyle='-',color = 'k')
      line2, =plt.plot(x_axis,non_injur, label="non_injur",linestyle='-',color = 'g')
      line3, =plt.plot(x_axis,injur, label="injur",linestyle='-',color = 'y')
      line4, =plt.plot(x_axis,fatal, label="fatal",linestyle='-',color = 'r')

      line5, =plt.plot(x_axis,train_overall, label="train_overall",linestyle='-.',color = 'k')
      line6, =plt.plot(x_axis,train_non_injur, label="train_non_injur",linestyle='-.',color = 'g')
      line7, =plt.plot(x_axis,train_injur, label="train_injur",linestyle='-.',color = 'y')
      line8, =plt.plot(x_axis,train_fatal, label="train_fatal",linestyle='-.',color = 'r')
      plt.legend()
      plt.xticks(np.arange(1, max_dep+1, 1.0))
      plt.title('Max Depth: {}, criterion: {}, splitter: {}'.format(max_dep+1,crt,split))
      plt.show();
  return result, all_model

def build_model_tree(X_train,y_train,test_x,test_y,max_dep,split_type,crt_type,classweight):
      result, all_model = test_parameter_tree(X_train, y_train, test_x, test_y, max_dep, split_type, crt_type, classweight)
      csv_name = 'tree_maxdep_{}_splittype_{}_crttype_{}_classweight_{}_accuracy.csv'.format(max_dep,split_type,crt_type,classweight)
      orders = ['Accuracy']  + [col for col in result if col != 'Accuracy']
      result = result[orders]
      result.to_csv(csv_name)
      return result, all_model

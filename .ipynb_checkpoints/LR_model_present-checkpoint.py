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
def model_result_present(modellr,train_x,train_y,test_x,test_y):
  modellr.fit(train_x, train_y)
  print("Logistic regression parameter",modellr.get_params())
  cm_test = confusion_matrix(test_y, modellr.predict(test_x))
  cm_train = confusion_matrix(train_y, modellr.predict(train_x))
  print ("Logistic regression Train Accuracy : ", modellr.score(train_x,train_y),'\n')
  print ("Logistic regression Test Accuracy : ", modellr.score(test_x,test_y),'\n')
  return cm_test,cm_train, modellr


def test_parameter(X_train,y_train,test_x,test_y, C, penalty,classweight,multiclass,solvertype):
  print('1：No Injury\n 2:Injury\n 3:Fatal\n')
  l = ['p','c','sev_T1_P1','sev_T2_P2','sev_T3_P3','sev_T1_P2','sev_T1_P3','sev_T2_P1','sev_T2_P3','sev_T3_P1','sev_T3_P2','Accuracy']
  result =pd.DataFrame(columns=l)
  # C =np.append([1.0],np.logspace(-4, 4, 30))
  # C = [1.0]
  # penalty = ['l1', 'l2']
  all_model = []
  i = 0
  for C_test in C:
    for p in penalty:
      print('p:  {}, C={}  '.format(p,C_test))
      lr_temp = linear_model.LogisticRegression(C=C_test, class_weight=classweight,multi_class=multiclass, penalty = p,solver=solvertype)
      # lr_temp = linear_model.LogisticRegression(C=C_test, class_weight="balanced",multi_class='multinomial', penalty = p,solver='saga')
      cm_test, cm_train, model = model_result_present(lr_temp,X_train,y_train,test_x,test_y)
      rows1 = {'p':p,'c':C_test,'sev_T1_P1':cm_test[0,0]/np.sum(cm_test[0]),'sev_T2_P2':cm_test[1,1]/np.sum(cm_test[1]),'sev_T3_P3':cm_test[2,2]/np.sum(cm_test[2]),'sev_T1_P2':cm_test[0,1]/np.sum(cm_test[0]),'sev_T1_P3':cm_test[0,2]/np.sum(cm_test[0]),'sev_T2_P1':cm_test[1,0]/np.sum(cm_test[1]),'sev_T2_P3':cm_test[1,2]/np.sum(cm_test[1]),'sev_T3_P1':cm_test[2,0]/np.sum(cm_test[2]),'sev_T3_P2':cm_test[2,1]/np.sum(cm_test[2]),'Accuracy':np.trace(cm_test)/cm_test.sum()}
      result = result.append(pd.DataFrame(rows1, index = ['test {}'.format(i)]))
      rows2 = {'p':p,'c':C_test,'sev_T1_P1':cm_train[0,0]/np.sum(cm_train[0]),'sev_T2_P2':cm_train[1,1]/np.sum(cm_train[1]),'sev_T3_P3':cm_train[2,2]/np.sum(cm_train[2]),'sev_T1_P2':cm_train[0,1]/np.sum(cm_train[0]),'sev_T1_P3':cm_train[0,2]/np.sum(cm_train[0]),'sev_T2_P1':cm_train[1,0]/np.sum(cm_train[1]),'sev_T2_P3':cm_train[1,2]/np.sum(cm_train[1]),'sev_T3_P1':cm_train[2,0]/np.sum(cm_train[2]),'sev_T3_P2':cm_train[2,1]/np.sum(cm_train[2]),'Accuracy':np.trace(cm_train)/cm_train.sum()}
      result = result.append(pd.DataFrame(rows2, index = ['train {}'.format(i)]))
      all_model.append(model)
      i+=1
  return result, all_model


def test_parameter_CV(X_train,y_train,test_x,test_y,cv,C,penalty,classweight,multiclass,solvertype):
  print('1：No Injury\n 2:Injury\n 3:Fatal\n')
  l = ['p','c','cv','sev_T1_P1','sev_T2_P2','sev_T3_P3','sev_T1_P2','sev_T1_P3','sev_T2_P1','sev_T2_P3','sev_T3_P1','sev_T3_P2','Accuracy']
  result =pd.DataFrame(columns=l)
  # cv =[3,5]
  # C = range(10,30,10)
  # penalty = [0,0.5,1.0]
  all_model = []
  i = 0
  for C_test in C:
    # print('p:  {}, C={}'.format(p,1.0))
    # lr_temp = linear_model.LogisticRegression(C=1.0, class_weight="balanced",multi_class='multinomial', penalty = p,solver='saga')
    # lr_temp = model_result_present(lr_temp,X_train,y_train,test_x,test_y)
    for p in penalty:
      for cv_value in cv:
        print('p:  {}, C={}  ,cv={}'.format(p,C_test,cv_value))
        lr_temp = linear_model.LogisticRegressionCV(Cs=C_test, class_weight=classweight,multi_class=multiclass, l1_ratios = p,solver=solvertype)
        # lr_temp = linear_model.LogisticRegressionCV(Cs=C_test, class_weight="balanced",multi_class='multinomial', l1_ratios = p,solver='saga')
        cm_test, cm_train, model = model_result_present(lr_temp,X_train,y_train,test_x,test_y)
        rows1 = {'p':p,'c':C_test,'cv':cv_value,'sev_T1_P1':cm_test[0,0]/np.sum(cm_test[0]),'sev_T2_P2':cm_test[1,1]/np.sum(cm_test[1]),'sev_T3_P3':cm_test[2,2]/np.sum(cm_test[2]),'sev_T1_P2':cm_test[0,1]/np.sum(cm_test[0]),'sev_T1_P3':cm_test[0,2]/np.sum(cm_test[0]),'sev_T2_P1':cm_test[1,0]/np.sum(cm_test[1]),'sev_T2_P3':cm_test[1,2]/np.sum(cm_test[1]),'sev_T3_P1':cm_test[2,0]/np.sum(cm_test[2]),'sev_T3_P2':cm_test[2,1]/np.sum(cm_test[2]),'Accuracy':np.trace(cm_test)/cm_test.sum()}
        result = result.append(pd.DataFrame(rows1, index = ['test {}'.format(i)]))
        rows2 = {'p':p,'c':C_test,'cv':cv_value,'sev_T1_P1':cm_train[0,0]/np.sum(cm_train[0]),'sev_T2_P2':cm_train[1,1]/np.sum(cm_train[1]),'sev_T3_P3':cm_train[2,2]/np.sum(cm_train[2]),'sev_T1_P2':cm_train[0,1]/np.sum(cm_train[0]),'sev_T1_P3':cm_train[0,2]/np.sum(cm_train[0]),'sev_T2_P1':cm_train[1,0]/np.sum(cm_train[1]),'sev_T2_P3':cm_train[1,2]/np.sum(cm_train[1]),'sev_T3_P1':cm_train[2,0]/np.sum(cm_train[2]),'sev_T3_P2':cm_train[2,1]/np.sum(cm_train[2]),'Accuracy':np.trace(cm_train)/cm_train.sum()}
        result = result.append(pd.DataFrame(rows2, index = ['train {}'.format(i)]))
        all_model.append(model)
        i+=1
  return result, all_model

def build_model_LR(X_train,y_train,test_x,test_y,C,penalty,classweight,multiclass,solvertype,cv=0):
      if cv == 0:
          result, all_model = test_parameter(X_train,y_train,test_x,test_y,C,penalty,classweight,multiclass,solvertype)
          csv_name = 'LR_pnlt_{}_cw_{}_mltcls_{}_svr_{}_mdlnum_{}_noncv_accuracy.csv'.format(penalty,classweight,multiclass,solvertype,len(all_model))
      else:
          result, all_model = test_parameter_CV(X_train,y_train,test_x,test_y,cv,C,penalty,classweight,multiclass,solvertype)
          csv_name = 'LRCV_pnlt_{}_cw_{}_mltcls_{}_svr_{}_mdlnum_{}_cv_{}_accuracy.csv'.format(penalty,classweight,multiclass,solvertype,len(all_model),cv)
      orders = ['Accuracy']  + [col for col in result if col != 'Accuracy']
      result = result[orders]
      result.to_csv(csv_name)
      return result, all_model

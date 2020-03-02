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
def deal_categorical_LabelEncoder(data_selected,dummy_fields):
    from sklearn.preprocessing import LabelEncoder
    lb_make = LabelEncoder()
    for each in dummy_fields:
      new_name = each+'_code'
      data_selected[new_name] = lb_make.fit_transform(data_selected[each])
    data_selected = data_selected.drop(columns=dummy_fields)
    return data_selected

def deal_categorical_dummy(data_selected, dummy_fields):
    # get_dummies处理数据，参数prefix是指处理之后数据的前缀
    # 例如mnth共有12个值，处理之后属性mnth将会被分解成12个属性，每个属性用0或者1表示
    # mnth将被分解为mnth_1, mnth_2, ..., mnth_12
    # 原本mnth=1的情况将变成 mnth_1 = 1，其余mnth_2,...,mnth_12都是0
    for each in dummy_fields:
      dummies = pd.get_dummies( data_selected.loc[:, each], prefix=each )
      data_selected = pd.concat( [data_selected, dummies], axis = 1 )
    data_selected = data_selected.drop(columns=dummy_fields)
    return data_selected

def create_data_for_ML(data, all_headers, dummy_headers, Regression):
    '''
    input: data cleaned data except for dummy or label Encoder
    all_header: List of columns wanted
    dummy_headers = required for dummy or label Encoder columns
    Regression: string either 'lr' for logistic Regression or 'tree'
    '''
    data_selected = data.filter(all_headers, axis=1)
    data_selected = data_selected.astype(int)
    #data_selected.info()
    if Regression == 'lr':
        data_selected = deal_categorical_dummy(data_selected, dummy_headers)
        SEV = data_selected.pop('P_ISEV')
        data_selected = pd.concat( [data_selected, SEV], axis = 1 )
    elif Regression == 'tree':
        data_selected = deal_categorical_LabelEncoder(data_selected, dummy_headers)
        SEV = data_selected.pop('P_ISEV')
        data_selected = pd.concat( [data_selected, SEV], axis = 1 )
    return data_selected


##################################################################################
def train_test_data(data_selected,time,testsize,year,interval,traffic_data_headers):
  '''generate training and testing data set
  input:
  data_selected pd.DateFrame
  time = 'True' use 'year' for testing, <year for training; 'False' then train_test split
  '''
  if time:
    train_data = data_selected[(data_selected.C_YEAR< year)&(data_selected.C_YEAR>=year-interval)]
    train_data = train_data.sample(frac=1)
    train_x = train_data.loc[:, train_data.columns != 'P_ISEV']
    train_x = train_x.drop(columns=["C_CASE"])
    train_y= train_data[traffic_data_headers[-1]]
    test_data = data_selected[(data_selected.C_YEAR == year)]
    test_x = test_data.loc[:, test_data.columns != 'P_ISEV']
    test_x = test_x.drop(columns=["C_CASE"])
    test_y= test_data[traffic_data_headers[-1]]
  else:
    train_data = data_selected.sample(frac=1)
    y = train_data[traffic_data_headers[-1]]
    train_x, test_x, train_y, test_y = train_test_split(train_data, y, test_size=testsize, random_state=42)
    test_data = test_x.copy()
    train_x = train_x.drop(columns=["P_ISEV"])
    test_x = test_x.drop(columns=["P_ISEV"])
    test_x = test_x.drop(columns=["C_CASE"])
    train_x = train_x.drop(columns=["C_CASE"])

  std = StandardScaler()
  train_x = std.fit_transform(train_x)
  test_x = std.transform(test_x)

  return test_data, train_x, test_x, train_y, test_y

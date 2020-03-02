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

def Cost_Per_Accident(output,sum_col,DC_Victim,DC_Accident):
  output_grouped = output.groupby('C_CASE')
  Acc_Cost = pd.DataFrame()
  Acc_Cost['Acc_Involve'] = output_grouped['C_CASE'].count()
  for col in sum_col:
    new_col = col+'_Acc' # means per accident
    Acc_Cost[new_col] = output_grouped[col].sum()#adding all victims in single accident DC together 

  Acc_Cost['C_ISEV_REAL'] = output_grouped['P_ISEV_REAL'].max()
  Acc_Cost['C_ISEV_PREDICT'] = output_grouped['P_ISEV_PREDICT'].max()
  Acc_Cost['Prob_Non_Fatal_Acc'] = output_grouped['Prob_Non_Fatal'].prod()
  Acc_Cost["Prob_PDO_Acc"] = output_grouped['Prob_PDO'].prod()
  Acc_Cost['Prob_Fatal_Acc'] = 1 - Acc_Cost.Prob_Non_Fatal_Acc
  Acc_Cost['Prob_Injur_Acc'] = 1 - Acc_Cost.Prob_PDO_Acc - Acc_Cost.Prob_Fatal_Acc
  Acc_Cost["Accident_type_Cost_Predict_EXP"] = DC_Accident['Fatal']*Acc_Cost.Prob_Fatal_Acc + DC_Accident['Injured']*Acc_Cost.Prob_Injur_Acc + DC_Accident['PDO']*Acc_Cost.Prob_PDO_Acc
  Acc_Cost["Accident_type_Cost_Predict"] = DC_Accident['Fatal']*((Acc_Cost.C_ISEV_PREDICT-3)==0).astype(int) + DC_Accident['Injured']*((Acc_Cost.C_ISEV_PREDICT-2)==0).astype(int) + DC_Accident['PDO']*((Acc_Cost.C_ISEV_PREDICT-1)==0).astype(int)
  Acc_Cost["Accident_type_Cost_Real"] = DC_Accident['Fatal']*((Acc_Cost.C_ISEV_REAL-3)==0).astype(int) + DC_Accident['Injured']*((Acc_Cost.C_ISEV_REAL-2)==0).astype(int) + DC_Accident['PDO']*((Acc_Cost.C_ISEV_REAL-1)==0).astype(int)
  Acc_Cost["Accident_Predict_Total"] = Acc_Cost.Accident_type_Cost_Predict + Acc_Cost.Predict_DC_Victim_Acc
  Acc_Cost["Accident_Predict_EXP_Total"] = Acc_Cost.Accident_type_Cost_Predict_EXP + Acc_Cost.Predict_DC_Victim_EXP_Acc
  Acc_Cost["Accident_Real_Total"] = Acc_Cost.Accident_type_Cost_Real + Acc_Cost.Real_DC_Victim_Acc
  return Acc_Cost


def error_analysis(Acc_Cost):
  error = pd.DataFrame()
  error["Predict_Total_Error"] = ((Acc_Cost.Accident_Predict_Total - Acc_Cost.Accident_Real_Total)/Acc_Cost.Accident_Real_Total)
  error["Predict_EXP_Total_Error"] = ((Acc_Cost.Accident_Predict_EXP_Total - Acc_Cost.Accident_Real_Total)/Acc_Cost.Accident_Real_Total)
  
  error["Predict_Accident_type_Error"] = ((Acc_Cost.Accident_type_Cost_Predict - Acc_Cost.Accident_type_Cost_Real)/Acc_Cost.Accident_type_Cost_Real)
  error["Predict_Accident_type_EXP_Error"] = ((Acc_Cost.Accident_type_Cost_Predict_EXP - Acc_Cost.Accident_type_Cost_Real)/Acc_Cost.Accident_type_Cost_Real)
  
  error["Predict_Acc_Victim_DC_Error"] = ((Acc_Cost.Predict_DC_Victim_Acc - Acc_Cost.Real_DC_Victim_Acc)/Acc_Cost.Real_DC_Victim_Acc)
  error["Predict_Acc_Victim_DC_EXP_Error"] = ((Acc_Cost.Predict_DC_Victim_EXP_Acc - Acc_Cost.Real_DC_Victim_Acc)/Acc_Cost.Real_DC_Victim_Acc)
  
  error["Acc_Predict_EXP_Prob_PDO_Error"] = Acc_Cost.Prob_PDO_Acc - Acc_Cost.Real_PDO_Prob_Acc
  error["Acc_Predict_Absolute_Prob_PDO_Error"] = Acc_Cost.Absolute_PDO_Prob_Acc - Acc_Cost.Real_PDO_Prob_Acc
  error["Acc_Predict_EXP_Prob_Injury_Error"] = Acc_Cost.Prob_Injury_Acc - Acc_Cost.Real_Injury_Prob_Acc
  error["Acc_Predict_Absolute_Prob_Injury_Error"] = Acc_Cost.Absolute_Injury_Prob_Acc - Acc_Cost.Real_Injury_Prob_Acc
  error["Acc_Predict_EXP_Prob_Fatal_Error"] = Acc_Cost.Prob_Fatal_Acc - Acc_Cost.Real_Fatal_Prob_Acc
  error["Acc_Predict_Absolute_Prob_Fatal_Error"] = Acc_Cost.Absolute_Fatal_Prob_Acc - Acc_Cost.Real_Fatal_Prob_Acc
    
  cm_accident_type = confusion_matrix(Acc_Cost['C_ISEV_REAL'], Acc_Cost['C_ISEV_PREDICT'])
  error['Accident_PDO_Accuracy'] = (cm_accident_type[0,0]/np.sum(cm_accident_type[0]))**2
  error['Accident_Injur_Accuracy'] = (cm_accident_type[1,1]/np.sum(cm_accident_type[1]))**2
  error['Accident_Fatal_Accuracy'] = (cm_accident_type[2,2]/np.sum(cm_accident_type[2]))**2
  error['Accident_Total_Accuracy'] = (np.trace(cm_accident_type)/cm_accident_type.sum())**2
  error['Accident_Type_Error'] = Acc_Cost['C_ISEV_PREDICT'] - Acc_Cost['C_ISEV_REAL']
  return error


def get_output_prob(test_data,test_x,model,url_path):
  '''Probability of all severity and final predicted severity result
  output: DataFrame with probability
  '''
  output = pd.DataFrame(test_data.C_CASE)
  output['P_ISEV_REAL'] = test_data['P_ISEV']
  test_prob = model.predict_proba(test_x)
  output['Prob_PDO'] = test_prob[:,0]
  output['Real_PDO_Prob'] = 0
  output.loc[output['P_ISEV_REAL'] == 1, 'Real_PDO_Prob'] = 1 #Actual probability of victim PDO
    
  output['Prob_Injury'] = test_prob[:,1]
  output['Real_Injury_Prob'] = 0
  output.loc[output['P_ISEV_REAL'] == 2, 'Real_Injury_Prob'] = 1 #Actual probability of victim Injury

  output['Prob_Fatal'] = test_prob[:,2]
  output['Real_Fatal_Prob'] = 0
  output.loc[output['P_ISEV_REAL'] == 3, 'Real_Fatal_Prob'] = 1 #Actual probability of victim Fatal
    
  output['Prob_Non_Fatal'] = 1 - output.Prob_Fatal

  output['P_ISEV_PREDICT'] = model.predict(test_x)
  output['Absolute_PDO_Prob'] = 0
  output.loc[output['P_ISEV_PREDICT'] == 1, 'Absolute_PDO_Prob'] = 1 #If using predicted category directly, probability of victim PDO
    
  output['Absolute_Injury_Prob'] = 0
  output.loc[output['P_ISEV_PREDICT'] == 2, 'Absolute_Injury_Prob'] = 1#If using predicted category directly, probability of victim PDO
    
  output['Absolute_Fatal_Prob'] = 0
  output.loc[output['P_ISEV_PREDICT'] == 3, 'Absolute_Fatal_Prob'] = 1#If using predicted category directly, probability of victim PDO   
    
  return output

def cost_prediction_money(test_data,test_x,model,url_path):
  
  cost = pd.read_csv(url_path)
  output = get_output_prob(test_data,test_x,model,url_path)
  num = lambda a: float(a.replace(',',''))
  DC_Victim = {'Fatal':float(cost.loc[15,'Fatal']),'Injured':float(cost.loc[15,'Injured']),'PDO':float(cost.loc[15,'Property Damage Only'])}
  DC_Accident = {'Fatal':num(cost.loc[32,'Fatal']),'Injured':num(cost.loc[32,'Injured']),'PDO':num(cost.loc[32,'Property Damage Only'])}
  output['Predict_DC_Victim'] = DC_Victim['Fatal']*((output.P_ISEV_PREDICT-3)==0).astype(int) + DC_Victim['Injured']*((output.P_ISEV_PREDICT-2)==0).astype(int) + DC_Victim['PDO']*((output.P_ISEV_PREDICT-1)==0).astype(int)
  output['Predict_DC_Victim_EXP'] = DC_Victim['Fatal']*output.Prob_Fatal + DC_Victim['Injured']*output.Prob_Injury + DC_Victim['PDO']*output.Prob_PDO
  output['Real_DC_Victim'] = DC_Victim['Fatal']*((output.P_ISEV_REAL-3)==0).astype(int) + DC_Victim['Injured']*((output.P_ISEV_REAL-2)==0).astype(int) + DC_Victim['PDO']*((output.P_ISEV_REAL-1)==0).astype(int)
  
  sum_col = ['Predict_DC_Victim','Predict_DC_Victim_EXP','Real_DC_Victim','Prob_PDO','Prob_Injury','Prob_Fatal','Absolute_PDO_Prob','Absolute_Injury_Prob','Absolute_Fatal_Prob','Real_PDO_Prob','Real_Injury_Prob','Real_Fatal_Prob']
  Acc_Cost = Cost_Per_Accident(output,sum_col,DC_Victim,DC_Accident)
  Acc_error = error_analysis(Acc_Cost)

  victim_error = pd.DataFrame()
  victim_error['Prob_EXP_error_PDO'] = output['Prob_PDO'] - output['Real_PDO_Prob']
  victim_error['Prob_Absolute_error_PDO'] = output['Absolute_PDO_Prob'] - output['Real_PDO_Prob']
    
  victim_error['Prob_EXP_error_Injury'] = output['Prob_Injury'] - output['Real_Injury_Prob']
  victim_error['Prob_Absolute_error_Injury'] = output['Absolute_Injury_Prob'] - output['Real_Injury_Prob']
    
  victim_error['Prob_EXP_error_Fatal'] = output['Prob_Fatal'] - output['Real_Fatal_Prob']
  victim_error['Prob_Absolute_error_Fatal'] = output['Absolute_Fatal_Prob'] - output['Real_Fatal_Prob']    
  
  victim_error['Predict_EXP_DC_error'] = (output['Predict_DC_Victim_EXP'] - output['Real_DC_Victim'])/output['Real_DC_Victim']
  victim_error['Predict_Absolute_DC_error'] = (output['Predict_DC_Victim'] - output['Real_DC_Victim'])/output['Real_DC_Victim']

  #Acc_Cost.to_csv('Acc_Cost.csv')
  return Acc_Cost,Acc_error,victim_error


def compare_model(test_data, test_x,model, url_path):
  allmodel_Acc_error_mean = pd.DataFrame()
  allmodel_Acc_error_mean_sqrt = pd.DataFrame()
  allmodel_victim_error_mean = pd.DataFrame()
  allmodel_victim_error_mean_sqrt = pd.DataFrame()
  # all_error[]
  i = 0
  Accident_Type_Error = []
  for m in model:
    Acc_Cost,Acc_error,victim_error = cost_prediction_money(test_data,test_x,m,url_path)
    print('\nmodel {}: \n'.format(i),m.get_params())
    Acc_error_sqr = Acc_error**2
    victim_error_sqr = victim_error**2
    allmodel_Acc_error_mean_sqrt[str(i)] = np.sqrt(Acc_error_sqr.mean())
    allmodel_victim_error_mean_sqrt[str(i)] = np.sqrt(victim_error_sqr.mean())
    allmodel_Acc_error_mean[str(i)] = Acc_error.mean()
    allmodel_victim_error_mean[str(i)] = victim_error.mean()
    #all_error[str(i)] = np.sqrt(error.loc[:, error.columns != 'Accident_Type_Error'].mean())
    Accident_Type_Error.append(Acc_error['Accident_Type_Error'].mean())
    i+=1
  #all_error.append(pd.DataFrame(Accident_Type_Error, index=['Accident_Type_Error']))
  return allmodel_Acc_error_mean_sqrt,allmodel_victim_error_mean_sqrt,allmodel_Acc_error_mean,allmodel_victim_error_mean
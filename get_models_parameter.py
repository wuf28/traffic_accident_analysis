import pandas as pd
def get_models_parameter(all_model,regression,test_data):
    if regression == 'lr':
        coeff_table = pd.DataFrame()
        i=0
        for model in all_model:
            coeff = test_data.columns.copy()
            coeff = coeff.to_list()
            coeff.remove('C_CASE')
            coeff.remove('P_ISEV')
            coeff_data = pd.DataFrame(model.coef_,columns = coeff,index=['PDO_coef','Injury_coef','Fatal_coef'])
            coeff_data.insert(0, 'intercept', model.intercept_)
            coeff_data.insert(0, 'model_number', [i,i,i])
            coeff_table = coeff_table.append(coeff_data)
            i+=1
        coeff_table.to_csv('coeff_data.csv')
import pandas as pd
def data_clean_remove_case(data_input,col,symbol_remove,row_only=0):
    '''
    input:
    data_input: the uncleaned pd.DataFrame
    col: the column name need to cleaned for not full data eg:'V_ID'
    symbol_remove: list of symbols indicating the line is defective
    row_only: if row_only = 1 then defects in single column only delete current row,
              if row_only = 0 then defects in single column only delete entire accidents.
              default =0, delete entire case
    return:
    a single column cleaned pd.DataFrame
    '''
    if row_only == 0:
        a = (data_input[col] == symbol_remove[0])
        for s in symbol_remove:
            a = a|(data_input[col] == s)
        data_delet = data_input[a]
        case_delet = list(dict.fromkeys(data_delet.C_CASE.tolist()))
        data_removed = data_input[~data_input['C_CASE'].isin(case_delet)]
    elif row_only == 1:
        a = (data_input[col] == symbol_remove[0])
        for s in symbol_remove:
            a = a|(data_input[col] == s)
        data_removed = data_input[~a]
    return data_removed

def data_parse(data_input,col,sel):
    '''
    input: data_input is unfiltered
    col: column to be filtered
    sel:
    '''
    a = (data_input[col]==sel[0])
    for s in sel:
      a = a | (data_input[col]==s)
    data_output = data_input[a]
    return data_output


def data_clean_columns(data_input,columns, v_num, symbol_remove=['U','N','Q','UU','QQ','XX','NN','UUUU','XXXX','NNNN'],row_only=0):
    '''
        input:
        data_input: the uncleaned pd.DataFrame
        columns: the list of column names need to cleaned for defective data (EXCEPT C_VEHS number of vehicles involced)eg:'V_ID'
        symbol_remove: list of symbols indicating the line is defective
        row_only: if row_only = 1 then defects in single column only delete current row,
                  if row_only = 0 then defects in single column only delete entire accidents.
                  default =0, delete entire case
        v_num: list of number of vehicles involved accidents to be filtered eg:[2] or [1] or [1,2]
        return:
        a cleaned pd.DataFrame
    '''
    data_input = data_parse(data_input,"C_VEHS",v_num)
    for col in columns:
        data_output = data_clean_remove_case(data_input,col,symbol_remove,row_only)
        data_input = data_output
    return data_input

def get_vihecle_year_category(data_input):
    data_input['V_YEAR'] = data_input['V_YEAR'].astype(int)
    data_input['V_AGE'] = data_input['C_YEAR'].sub(data_input['V_YEAR'], axis = 0)
    data_delet = data_input[(data_input.V_AGE< -1)]
    case_delet = data_delet.C_CASE.tolist()
    temp = list(dict.fromkeys(case_delet))
    data_input = data_input[~data_input['C_CASE'].isin(temp)]
    bins = range(data_input['V_AGE'].min(),data_input['V_AGE'].max()+(5-data_input['V_AGE'].max()%5),5)
    labels = range(1,len(bins))
    data_input['V_AGE_GRP'] = pd.cut(data_input.V_AGE,bins, labels=labels,include_lowest=True, right=True)
    data_input['V_AGE_GRP'] = data_input['V_AGE_GRP'].astype(int)

    return data_input


def get_person_year_category(data_input):
    data_input['P_AGE'] = data_input['P_AGE'].astype(int)
    bins = [data_input['P_AGE'].min(),4,15,19,24,34,44,54,64,data_input['P_AGE'].max()]
    #bins = [data_input['P_AGE'].min(),data_input['P_AGE'].max(),200]
    labels = range(1,len(bins))
    data_input['P_AGE_GRP'] = pd.cut(data_input.P_AGE,bins, labels=labels, include_lowest=True)
    data_input['P_AGE_GRP'] = data_input['P_AGE_GRP'].astype(int)

    return data_input

def modify_data(data_input,columns):

    for col in columns:
        if col == 'V_YEAR':
            data_input = get_vihecle_year_category(data_input)
        elif col == 'P_SEX':
            data_input.loc[data_input['P_SEX'] == 'M', 'P_SEX'] = 1
            data_input.loc[data_input['P_SEX'] == 'F', 'P_SEX'] = 0
        elif col == 'P_AGE':
            data_input = get_person_year_category(data_input)

    return data_input

def get_data_stats(data_input,columns):
    for col in columns:
        print('{} column data is distributed as below: \n'.format(col))
        print(data_input[col].value_counts(normalize=True))
        print('Total size : {}'.format(data_input[col].shape))

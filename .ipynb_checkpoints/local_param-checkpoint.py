import pandas as pd
import numpy as np
csv_file   = 'NCDB_1999_to_2017.csv' # y_start    = 1999 #inclusive # y_end      = 2017 #exlusive
money_path = 'Traffic Accident Cost Calculation.csv'
row_only   = 0                      # 0 means delete entire traffic case, 0 means remove single row only
regression = 'lr'                   # 'tree' or 'lr'  single string only
time       = True                   # True: using latest year as test and previous as training; False: Random Test Train Split
test_size  = 0.2                    # test size if time = False
year       = 2017                   # year for test and earlier for training
interval   = 2017-1999              # m-n; training set intervals: start from n year, to m-1 year
classweight= "balanced"             # { 1:1, 2:1, 3: 70 } or "balanced": w1n1+w1n2+w3n3 = n1+n2+n3 & w1n1=w1n2=w3n3
C          = [1.0]     #coefficient of panelty for regularization; C = np.append([1.0],np.logspace(-4, 4, 30)) for best C search
#C          = [0.0001]
C = [10]                            # Cross Validation C
# penalty    = ['l1','l2']           # ['l1'] for lasso, ['l2'] for Ridge
penalty = [0,0.5,1.0]           # penalty = [0,0.5,1.0] # if using [‘elasticnet’] combination of l1(=1) and l2(=0),0.5 half l1 and half l2
solver     ='saga' 
                                    #‘newton-cg’, ‘sag’, ‘saga’ and ‘lbfgs’ handle multinomial loss
                                    #‘liblinear’(not support setting penalty='none') and ‘saga’ also handle L1 penalty
                                    #‘newton-cg’, ‘lbfgs’, ‘sag’ and ‘saga’ handle L2 or no penalty
                                    #‘saga’ also supports ‘elasticnet’ penalty
multi_class='multinomial'           # {‘auto’, ‘ovr’, ‘multinomial’}
# cv = 0                              # 0 means no cross validation, [3,5] 3 folds and 5 folds
cv =[5]
C = [10]
# C = range(10,30,10)              #C selection in cross validation
max_dep = 40                        # max_depth of tree
#split_type = ["best","random"]
split_type = ["best"]               #["best","random"] split type 
#crt_type = ["gini","entropy"]      # criterion type
crt_type = ["gini"]
v_num         = [1,2]                 #number of vihecles to include
symbol_remove = ['U','N','Q','UU','QQ','XX','NN','UUUU','XXXX','NNNN']
columns_clean = ['C_HOUR','C_MNTH','C_WDAY','C_VEHS','P_ISEV','C_CONF','C_RCFG','V_TYPE','V_YEAR','P_AGE','P_PSN','P_SEX','P_ID','P_SAFE','P_USER','C_RSUR','C_RALN','C_WTHR']
columns_grp = ['C_HOUR','C_MNTH','C_WDAY','V_YEAR','P_SEX','P_AGE']
traffic_data_headers = ['C_CASE','C_YEAR','C_VEHS','C_HOUR','C_MNTH','C_WDAY','C_RSUR','C_RALN','C_WTHR','C_CONF','C_RCFG','V_TYPE','P_SEX','V_AGE_GRP','P_AGE_GRP','P_PSN',"P_SAFE",'P_USER','P_ISEV']
columns_stats = traffic_data_headers.copy()
columns_stats.remove('C_CASE')
# columns_stats = ['C_YEAR','C_VEHS','C_RSUR','C_RALN','C_WTHR','C_CONF','C_RCFG','V_TYPE','P_SEX','V_AGE_GRP','P_AGE_GRP','P_PSN',"P_SAFE",'P_USER','P_ISEV']
dummy_fields = ['C_HOUR','C_MNTH','C_RSUR','C_RALN','C_WTHR','C_CONF','C_RCFG','V_TYPE','P_PSN',"P_SAFE",'P_USER'] # created into dummy variables for logistic regression or Lable Encoder for tree

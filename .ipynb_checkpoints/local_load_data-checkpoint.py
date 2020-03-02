import pandas as pd
def start_load(csv_file):
    data_All_years = pd.read_csv(csv_file)
    print('\nTotal Population involed in all traffic accidents: ',len(data_All_years.C_CASE.tolist()))
    num_accidents = list(dict.fromkeys(data_All_years.C_CASE.tolist()))
    print('\nTotal number of all traffic accidents: ',len(num_accidents))
    data_All_years['C_SEV'].value_counts().plot(kind='bar')
    return data_All_years

def get_TrafficData_CSV_year(Full_Data,start,end):
      '''argument:
          Full_Data: pandas DataFrame
          start: start year
          end: end year
          return: a .csv file that contains year of data from start year(inclusivee) to end year(exclusive)
          name NCDB_start_to_end.csv
      '''
      #data = Full_Data[(Full_Data.C_YEAR< end) and (Full_Data.C_YEAR >= start)]
      data = Full_Data[(Full_Data['C_YEAR'] >= start) &  (Full_Data['C_YEAR'] < end)]
      name = 'NCDB_{}inc_to_{}exc.csv'.format(start,end)
      data.to_csv(name)
      return data

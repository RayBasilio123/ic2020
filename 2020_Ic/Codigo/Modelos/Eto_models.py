#building the tree
# Import the necessary modules and libraries
import numpy as np
import pandas as pd
from datetime import timedelta,datetime
import statsmodels.api as sm
import time
import math
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import f1_score
from Modelos.Eto_experiments import gera_serie,timer
from Modelos.Eto_experiments import get_x2
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from xgboost import XGBRegressor
from matplotlib import pyplot
from sklearn.model_selection import RandomizedSearchCV
from pprint import pprint
from sklearn.model_selection import GridSearchCV
import joblib
# df_patricia = pd.read_csv('C:/Users/Ray/Documents/2020_Ic/Dados/data_PEto.csv') 
# df_patricia
def run_arimaDay(series, steps_ahead, configuracao):
  result = []
  
  #Lista de data+hora que será previsto
  begin = series.index.max() + timedelta(days=0)
  date_list = [begin + timedelta(days=x) for x in range(1,steps_ahead+1)]
  
  #ores da série
  ues = series.ues

  #ARIMA
  start_fit = time.time()
  mod = sm.tsa.statespace.SARIMAX(ues, order=configuracao)
  res = mod.fit(disp=False)
  t_fit = time.time() - start_fit
  
  start_fcast = time.time() 
  forecast = res.forecast(steps=steps_ahead)
  t_fcast = time.time() - start_fcast 
  
  #Resultado no formato para ser exibido no gráfico
  for i in range(steps_ahead):
    if forecast[i] < 0: 
      result.append([date_list[i].strftime('%d/%m/%Y '),0])
    else:
      result.append([date_list[i].strftime('%d/%m/%Y '),round((forecast[i]),3)])

  return result, t_fit, t_fcast

def run_sarimaDay(series, steps_ahead, config_ordem, config_sazonal):
  result = []
  
  #Lista de data+hora que será previsto
  begin = series.index.max() + timedelta(days=0)
  date_list = [begin + timedelta(days=x) for x in range(1,steps_ahead+1)]
  
  #ores da série
  ues = series.ues

  #ARIMA
  mod = sm.tsa.statespace.SARIMAX(ues, order=config_ordem, seasonal_order=config_sazonal)
  res = mod.fit(disp=False)
  forecast = res.forecast(steps=steps_ahead)

  #Resultado no formato para ser exibido no gráfico
  for i in range(steps_ahead):
    if forecast[i] < 0: 
      result.append([date_list[i],0])
    else:
      result.append([date_list[i],round((forecast[i]),3)])

  return result


def arvore(df,lista, lags, Eto, lags_eto,variavel_Alvo):
  
  x1_train, x1_test,y1_train, y1_test = train_test(df, lista,lags,Eto,lags_eto,variavel_Alvo)
  # print("***************************************")
  # print(x1_train,"x1_train")
  # #[6569 rows x 10 columns] x1_train
  # print("*****************++*********************")
  # print(x1_test,"x1_test")
  # print("******************---********************")
  # print(y1_train,"y1_train")
  # print("****************1234*********************")
  # print(y1_test,"y1_test")
  # print("*****************asddfg******************")

  # Ajuste manual
  model = DecisionTreeRegressor(random_state = 42)
  
  # Ajuste por grid
  # model = DecisionTreeRegressor(random_state=42,max_depth= 5, max_features='log2', max_leaf_nodes= 10, min_samples_leaf=1, min_weight_fraction_leaf= 0.1, splitter='best')
 # 1 Dia
  # model = DecisionTreeRegressor(random_state=42,max_depth= 5, max_features='auto', max_leaf_nodes= None, min_samples_leaf=1, min_weight_fraction_leaf= 0.1, splitter='best')
  # model = DecisionTreeRegressor(random_state=42,max_depth= 5, max_features='auto', max_leaf_nodes= None, min_samples_leaf=1, min_weight_fraction_leaf= 0.1, splitter='best')
  # 3 Dias
  # model = DecisionTreeRegressor(random_state=42,max_depth= 5, max_features='auto', max_leaf_nodes= None, min_samples_leaf=1, min_weight_fraction_leaf= 0.1, splitter='best')
  # model = DecisionTreeRegressor(random_state=42,max_depth= 5, max_features='auto', max_leaf_nodes= None, min_samples_leaf=1, min_weight_fraction_leaf= 0.1, splitter='best')
  #7 Dias
  # model = DecisionTreeRegressor(random_state=42,max_depth= 5, max_features='auto', max_leaf_nodes= 10, min_samples_leaf=1, min_weight_fraction_leaf= 0.1, splitter='best')
  # model = DecisionTreeRegressor(random_state=42,max_depth= 5, max_features='log2', max_leaf_nodes= 10, min_samples_leaf=1, min_weight_fraction_leaf= 0.1, splitter='best')
  #10 Dias
  # model = DecisionTreeRegressor(random_state=42,max_depth= 5, max_features='auto', max_leaf_nodes= None, min_samples_leaf=1, min_weight_fraction_leaf= 0.1, splitter='best')
  model = DecisionTreeRegressor(random_state=42,max_depth= 5, max_features='log2', max_leaf_nodes= 10, min_samples_leaf=1, min_weight_fraction_leaf= 0.1, splitter='best')
  model.fit(x1_train, y1_train)                 
  print("------------------")
  print("Arvore")
  y1_pred = model.predict(x1_test)
  mse = mean_squared_error(y1_test, y1_pred)
  std_mse=np.std(np.sqrt((y1_pred - y1_test)**2))
  
  print("std_mse",round(std_mse,2))
  rmse = math.sqrt(mse)
  print("Erro medio absoluto----",round(mean_absolute_error(y1_test, y1_pred),2))
  
  # pyplot.plot(np.arange(y1_test.shape[0]),y1_test, label='Expected tree ')
  # pyplot.plot(y1_pred, label='Predicted tree')
  # pyplot.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
  #               mode="expand", borderaxespad=0, ncol=3)
  # pyplot.show()
  return lista,lags,Eto,lags_eto,round(rmse,2)

def arvores(df,arvore_parametros,variavel_Alvo):
  lista_colunas=["lista","lista_lags","Eto","lags_eto","rmse"]
  tb = pd.DataFrame(columns=lista_colunas)
  print("Arvores")
  for x in range(len(arvore_parametros)):
    a=arvore(df,arvore_parametros[x][0],arvore_parametros[x][1],arvore_parametros[x][2],arvore_parametros[x][3],variavel_Alvo)
 
    tb.loc[x,'lista']=a[0]
    tb.loc[x,'lista_lags']=a[1]
    tb.loc[x,'Eto']=a[2]
    tb.loc[x,'lags_eto']=a[3]
    tb.loc[x,"rmse"]=a[4]

  print(tb)

  return tb

def train_test(df, lista,lags,Eto,lags_eto,variavel_Alvo,data_Itest= '2013-01-01',data_Ftest= '2013-12-31',data_Itreino='1993-01-01',data_Ftreino='2011-12-31'):
  tabela = get_x2(df, lista,lags,Eto,lags_eto)
  selecao_treino = (tabela[0]['Data'] >= data_Itreino) & (tabela[0]['Data'] <= data_Ftreino)
  selecao_teste = (tabela[0]['Data'] >= data_Itest) & (tabela[0]['Data'] <= data_Ftest)

  x1_train = tabela[0][selecao_treino].drop("Data", axis=1)
  x1_test = tabela[0][selecao_teste].drop("Data", axis=1)
  
  y1_train = df[variavel_Alvo][selecao_treino]
  y1_test = df[variavel_Alvo][selecao_teste]

  x1_train = x1_train[tabela[1]:]
  y1_train = y1_train[tabela[1]:]

  
  print("***************************************")
  print(x1_train.shape,"x1_train",tabela[1],"Maxx")
  #[6569 rows x 10 columns] x1_train
  print("*****************++*********************")
  print(x1_test.shape,"x1_test")
  print("******************---********************")
  print(y1_train.shape,"y1_train")
  print("****************1234*********************")
  print(y1_test.shape,"y1_test")
  print("*****************asddfg******************")
 
           
  return x1_train, x1_test,y1_train, y1_test


def florestaAleatoria(df,lista, lags, Eto, lags_eto,variavel_Alvo):
  x1_train, x1_test,y1_train, y1_test = train_test(df, lista,lags,Eto,lags_eto,variavel_Alvo)
  # tabela = get_x2(df, lista,lags,Eto,lags_eto)
  
  # #Quando era em porcentagem eu usava assim :
  # # x1 = tabela[0].drop("Data", axis=1)
  # # y1 = df[variavel_Alvo]

  # # train_size = int(((len(tabela[0])) -tabela[1]) * 0.82)
  # # x1_train, x1_test = x1[tabela[1]:train_size], x1[train_size:int(len(x1))]
  # # y1_train, y1_test = y1[tabela[1]:train_size], y1[train_size:int(len(y1))]  
  # selecao_treino = (tabela[0]['Data'] >= '1993') & (tabela[0]['Data'] <= '2011-12-31')
  # selecao_teste = (tabela[0]['Data'] >= '2012-01-01') & (tabela[0]['Data'] <= '2012-12-31')

  # x1_train = tabela[0][selecao_treino].drop("Data", axis=1)
  # x1_test = tabela[0][selecao_teste].drop("Data", axis=1)
  
  # y1_train = df[variavel_Alvo][selecao_treino]
  # y1_test = df[variavel_Alvo][selecao_teste]
  # x1_train, x1_test = x1_train[tabela[1]:], x1_test
  # y1_train, y1_test = y1_train[tabela[1]:], y1_test

 

  # Ajuste manual
  # model = RandomForestRegressor(n_estimators=1000,random_state = 42 )

  # Ajuste por grid

 # 1 Dia
  # model = RandomForestRegressor(random_state = 42 ,bootstrap=True, max_depth= 8, max_features= 5, min_samples_leaf= 4, min_samples_split= 8, n_estimators=1000)
  # model = RandomForestRegressor(random_state = 42 ,bootstrap=True, max_depth= 11, max_features= 2, min_samples_leaf= 5, min_samples_split= 12, n_estimators=1000)
  # 3 Dias
  # model = RandomForestRegressor(random_state = 42 ,bootstrap=True, max_depth= 8, max_features= 4, min_samples_leaf= 5, min_samples_split= 8, n_estimators=1000)
  # model = RandomForestRegressor(random_state = 42 ,bootstrap=True, max_depth= 5, max_features= 3, min_samples_leaf= 4, min_samples_split= 8, n_estimators=100)
  #7 Dias
  # model = RandomForestRegressor(random_state = 42 ,bootstrap=True, max_depth= 8, max_features= 3, min_samples_leaf= 4, min_samples_split= 8, n_estimators=100)
  # model = RandomForestRegressor(random_state = 42 ,bootstrap=True, max_depth= 5, max_features= 3, min_samples_leaf= 5, min_samples_split= 12, n_estimators=100)
  #10 Dias
  # model = RandomForestRegressor(random_state = 42 ,bootstrap=True, max_depth= 8, max_features= 5, min_samples_leaf= 3, min_samples_split= 10, n_estimators=500)
  model = RandomForestRegressor(random_state = 42 ,bootstrap=True, max_depth= 5, max_features= 2, min_samples_leaf= 5, min_samples_split= 8, n_estimators=100)

  model.fit(x1_train, y1_train)
  print("------------------")
  print("floresta")
  y1_pred = model.predict(x1_test)
  mse = mean_squared_error(y1_test, y1_pred)
  std_mse=np.std(np.sqrt((y1_pred - y1_test)**2))
  
  print("std_mse",round(std_mse,2))
  rmse = math.sqrt(mse)
  print("Erro medio absoluto----",round(mean_absolute_error(y1_test, y1_pred),2))
  # pyplot.plot(np.arange(y1_test.shape[0]),y1_test, label='Expected Random forests ')
  # pyplot.plot(y1_pred, label='Predicted Random forests')
  # pyplot.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
  #               mode="expand", borderaxespad=0, ncol=3)
  # pyplot.show()
    
  return lista,lags,Eto,lags_eto,round(rmse,2),

def florestasAleatorias(df,arvore_parametros,variavel_Alvo):
  lista_colunas=["lista","lista_lags","Eto","lags_eto","rmse"]
  tb = pd.DataFrame(columns=lista_colunas)
  print("florestass")
  for x in range(len(arvore_parametros)):
    a=florestaAleatoria(df,arvore_parametros[x][0],arvore_parametros[x][1],arvore_parametros[x][2],arvore_parametros[x][3],variavel_Alvo)
 
    tb.loc[x,'lista']=a[0]
    tb.loc[x,'lista_lags']=a[1]
    tb.loc[x,'Eto']=a[2]
    tb.loc[x,'lags_eto']=a[3]
    tb.loc[x,"rmse"]=a[4]
    joblib.dump(a,"floresta")

  print(tb)

  return tb


   
  

def xgb(df,lista, lags, Eto, lags_eto,variavel_Alvo):
  
  x1_train, x1_test,y1_train, y1_test = train_test(df, lista,lags,Eto,lags_eto,variavel_Alvo)
  # tabela = get_x2(df, lista,lags,Eto,lags_eto)
  # x1 = tabela[0].drop("Data", axis=1)
  # y1 = df[variavel_Alvo]
  
  # train_size = int(((len(tabela[0])) -tabela[1]) * 0.82)
  # x1_train, x1_test = x1[tabela[1]:train_size], x1[train_size:int(len(x1))]
  # y1_train, y1_test = y1[tabela[1]:train_size], y1[train_size:int(len(y1))]

  # selecao_treino = (tabela[0]['Data'] >= '1993') & (tabela[0]['Data'] <= '2011-12-31')
  # selecao_teste = (tabela[0]['Data'] >= '2013-01-01') & (tabela[0]['Data'] <= '2013-12-31')

  # x1_train = tabela[0][selecao_treino].drop("Data", axis=1)
  # x1_test = tabela[0][selecao_teste].drop("Data", axis=1)
  
  # y1_train = df[variavel_Alvo][selecao_treino]
  # y1_test = df[variavel_Alvo][selecao_teste]
  # x1_train, x1_test = x1_train[tabela[1]:], x1_test
  # y1_train, y1_test = y1_train[tabela[1]:], y1_test
  # Padrão 
  # model = XGBRegressor(random_state = 42)
  # Ajuste manual
  # model = XGBRegressor(objective='reg:squarederror', n_estimators=1000,random_state = 42)

  # Ajuste por grid
  
  # 1 Dia
  # model = XGBRegressor( random_state = 42,colsample_bytree= 1, gamma= 1.5, learning_rate= 0.07, max_depth= 3, min_child_weight= 5, subsample= 0.8)
  # model = XGBRegressor( random_state = 42,colsample_bytree= 0.6, gamma= 5, learning_rate= 0.07, max_depth= 4, min_child_weight= 1, subsample= 0.8)
  
  # 3 Dias
  # model = XGBRegressor( random_state = 42,colsample_bytree= 1, gamma= 0.5, learning_rate= 0.05, max_depth= 3, min_child_weight= 5, subsample= 0.8)
  # model = XGBRegressor( random_state = 42,colsample_bytree= 0.6, gamma= 5, learning_rate= 0.05, max_depth= 3, min_child_weight= 5, subsample= 0.6)
  
  #7 Dias
  # model = XGBRegressor( random_state = 42,colsample_bytree= 1, gamma= 5, learning_rate= 0.07, max_depth= 4, min_child_weight= 1, subsample= 1)
  # model = XGBRegressor( random_state = 42,colsample_bytree= 0.6, gamma= 5, learning_rate= 0.05, max_depth= 3, min_child_weight= 5, subsample= 0.6)
  
  #10 Dias
  # model = XGBRegressor( random_state = 42,colsample_bytree= 0.6, gamma= 1, learning_rate= 0.05, max_depth= 4, min_child_weight= 10, subsample= 0.8)
  model = XGBRegressor( random_state = 42,colsample_bytree= 1, gamma= 5, learning_rate= 0.05, max_depth= 3, min_child_weight= 1, subsample= 1)
  model.fit(x1_train, y1_train)
  print("------------------")
  print("xgb")
  y1_pred = model.predict(x1_test)
  mse = mean_squared_error(y1_test, y1_pred)
  std_mse=np.std(np.sqrt((y1_pred - y1_test)**2))
  
  print("std_mse",round(std_mse,2))
  rmse = math.sqrt(mse)
  print("Erro medio absoluto----",round(mean_absolute_error(y1_test, y1_pred),2))
  # pyplot.plot(np.arange(y1_test.shape[0]),y1_test, label='Expected xgb ')
  # pyplot.plot(y1_pred, label='Predicted xgb')
  # pyplot.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
  #               mode="expand", borderaxespad=0, ncol=3)
  # pyplot.show()
    
  return lista,lags,Eto,lags_eto,round(rmse,2)

def xgbs(df,arvore_parametros,variavel_Alvo):
  lista_colunas=["lista","lista_lags","Eto","lags_eto","rmse"]
  tb = pd.DataFrame(columns=lista_colunas)
  print("xgbs")
  for x in range(len(arvore_parametros)):
    a=xgb(df,arvore_parametros[x][0],arvore_parametros[x][1],arvore_parametros[x][2],arvore_parametros[x][3],variavel_Alvo)
 
    tb.loc[x,'lista']=a[0]
    tb.loc[x,'lista_lags']=a[1]
    tb.loc[x,'Eto']=a[2]
    tb.loc[x,'lags_eto']=a[3]
    tb.loc[x,"rmse"]=a[4]

  print(tb)

  return tb





# gridSearch floresta
 
def RandomizedSearchF(df, lista,lags,Eto,lags_eto,variavel_Alvo):
 
 x1_train, x1_test,y1_train, y1_test = train_test(df, lista,lags,Eto,lags_eto,variavel_Alvo)

 # Number of trees in random forest
 n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
 # Number of features to consider at every split
 max_features = ['auto', 'sqrt']
 # Maximum number of levels in tree
 max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
 max_depth.append(None)
 # Minimum number of samples required to split a node
 min_samples_split = [2, 5, 10]
 # Minimum number of samples required at each leaf node
 min_samples_leaf = [1, 2, 4]
 # Method of selecting samples for training each tree
 bootstrap = [True, False]

 #  Create the random grid
 random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

 pprint(random_grid)

 # First create the base model to tune
 rf = RandomForestRegressor(random_state = 42)
 # Random search of parameters, using 3 fold cross validation, 
 # search across 100 different combinations, and use all available cores
 rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid,
                              n_iter = 100, scoring='neg_mean_absolute_error', 
                              cv = 3, verbose=2, random_state=42, n_jobs=-1,
                              return_train_score=True)

 # Fit the random search model
 rf_random.fit(x1_train, y1_train);

 return rf_random.best_params_

def RandomizedSearchsF(df,parametros,variavel_Alvo):
  tb = pd.DataFrame(columns=lista_colunas)
  print("RandomSearch")
  for x in range(len(parametros)):
    a=RandomizedSearchF(df,parametros[x][0],parametros[x][1],parametros[x][2],parametros[x][3],variavel_Alvo)
 
    tb.loc[x,'Best_ramdon_hyperparameter']=[a]
  print('---------------------------------------------------------------------------------')

  return tb


def gridSearchF(df, lista,lags,Eto,lags_eto,variavel_Alvo):
  x1_train, x1_test,y1_train, y1_test = train_test(df, lista,lags,Eto,lags_eto,variavel_Alvo)
   # Create the parameter grid based on the results of random search 
  param_grid = {
      'bootstrap': [True],
      'max_depth': [10,80, 90, 100, 110],
      'max_features': [2, 3],
      'min_samples_leaf': [3, 4, 5],
      'min_samples_split': [8, 10, 12],
      'n_estimators': [100, 200, 300, 1000]
  }

  # Create a base model
  rf = RandomForestRegressor(random_state = 42)

  # Instantiate the grid search model
  grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                            cv = 3, n_jobs = -1, verbose = 2, return_train_score=True)

  print(" -----------------grid_search.fit--------------")
 
  start_time=timer(None)

  grid_search.fit(x1_train, y1_train);

  timer(start_time)  
  
    
  return rf_random.best_params_

def gridSearchFs(df,parametros,variavel_Alvo):

  lista_colunas=["gridSearch"]
  tb = pd.DataFrame(columns=lista_colunas)
  
  print("gridSearchFs")
  for x in range(len(parametros)):
    a=gridSearchF(df,parametros[x][0],parametros[x][1],parametros[x][2],parametros[x][3],variavel_Alvo)
 
    tb.loc[x,'Best_ramdon_hyperparameter']=[a]
  print('---------------------------------------------------------------------------------')
  

  return tb

# gridSearch Arvore

def gridSearchA(df, lista,lags,Eto,lags_eto,variavel_Alvo):
  tb = pd.DataFrame()
  x1_train, x1_test,y1_train, y1_test = train_test(df, lista,lags,Eto,lags_eto,variavel_Alvo)
   # Create the parameter grid based on the results of random search 
  
  param_grid={"splitter":["best","random"],
            "max_depth" : [1,3,5,7,9,11,12],
            "min_samples_leaf":[1,2,3,4,5,6,7,8,9,10],
            "min_weight_fraction_leaf":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
            "max_features":["auto","log2","sqrt",None],
            "max_leaf_nodes":[None,10,20,30,40,50,60,70,80,90] 
            }
 
  # Create a base model
  rf = DecisionTreeRegressor(random_state = 42)

  # Instantiate the grid search model
  grid_search=GridSearchCV(rf,param_grid=param_grid,scoring='neg_mean_squared_error',cv=3,verbose=3,n_jobs = -1,return_train_score=True)
  

  print(" -----------------grid_search.fit--------------")
  start_time=timer(None)

  grid_search.fit(x1_train, y1_train);

  timer(start_time)  

  return grid_search.best_params_

def gridSearchAs(df,parametros,variavel_Alvo):
  print('---------------------------------------------------------------------------------')
  
  for x in range(len(parametros)):
    a=gridSearchA(df,parametros[x][0],parametros[x][1],parametros[x][2],parametros[x][3],variavel_Alvo)
    
    print(x,"Best_hyperparameter_Xgb", a)
   
  print('---------------------------------------------------------------------------------')
  

  return tb

# gridgridSearch Xgboost

def gridSearchXgb(df, lista,lags,Eto,lags_eto,variavel_Alvo):
  x1_train, x1_test,y1_train, y1_test = train_test(df, lista,lags,Eto,lags_eto,variavel_Alvo)
   # Create the parameter grid based on the results of random search 
  param_grid = {
        'min_child_weight': [1, 5, 10],
        # Define a soma mínima dos pesos de todas as observações exigidas em uma criança.
        # Isso é semelhante a min_child_leaf no GBM, mas não exatamente. Isso se refere à “soma dos pesos” mínimas das observações, enquanto o GBM tem “número mínimo de observações”.
        # Usado para controlar o sobreajuste.
        'gamma': [0.5, 1, 1.5, 2, 5],
        # Um nó é dividido apenas quando a divisão resultante dá uma redução positiva na função de perda.
        # Gama especifica a redução de perda mínima necessária para fazer uma divisão.
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        # Semelhante a max_features em GBM. Denota a fração de colunas a serem amostras 
        # aleatoriamente para cada árvore.Valores típicos: 0,5-1
        'max_depth': [3, 4, 5],
        # booster [default = gbtree]
        # Selecione o tipo de modelo a ser executado em cada iteração. Possui 2 opções:
        # gbtree: modelos baseados em árvore
        # gblinear: modelos lineares
        }

  # Create a base model
  rf = XGBRegressor(random_state = 42)

  # Instantiate the grid search model
  grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                            cv = 3, n_jobs = -1, verbose = 2, return_train_score=True)

  print(" -----------------grid_search___Xgb.fit--------------")
  grid_search.fit(x1_train, y1_train);
    
  return grid_search.best_params_

def gridSearchXgbs(df,parametros,variavel_Alvo):
  tb = pd.DataFrame()
  for x in range(len(parametros)):
    a=gridSearchXgb(df,parametros[x][0],parametros[x][1],parametros[x][2],parametros[x][3],variavel_Alvo)
 
    tb.loc[x,'Best_hyperparameter_Xgb']=[a]
    print(x,"--->",a)

  print('---------------------------------------------------------------------------------')
  return tb
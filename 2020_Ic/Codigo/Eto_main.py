import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
from Modelos.Eto_experiments import get_x2, get_x30,resource_ranking,column_Filter,list_Format,tab_to_tab_lags
from Tratamento.variaveis import latitude_2,  altitude_2, sigma, G, Gsc
from Tratamento.Eto_generator import gera_serie
from Modelos.Eto_models import arvores,florestasAleatorias,xgbs,RandomizedSearchsF,gridSearchFs,gridSearchXgbs,gridSearchAs
import os.path
import seaborn as sns 
import matplotlib as plt
import matplotlib.pyplot as plt

# from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
path= 'C:/Users/Ray/Documents/'

data_Ray = pd.read_csv(path+'IC/2020_Ic/Dados/Dados_PP_Eto.csv') 
eto_Ray = gera_serie(data_Ray, latitude_2, altitude_2, Gsc, sigma, G)
data_Ray["Eto"] = eto_Ray
atributeR = ["Eto", "temp_media", "temp_max","radiacao"]
resultado = get_x30(data_Ray, atributeR, ['Eto'])
if(os.path.exists(path+'IC/2020_Ic/Dados/Tabela30R_Lags.csv')):
  print("CSV Ray já existente !")
else:
  print("Exportando dados Ray ...")
  resultado[0].to_csv(path+'IC/2020_Ic/Dados/Tabela30R_Lags.csv')
  print("Concluido !")


data_Patricia_Eto = pd.read_csv(path+'IC/2020_Ic/Dados/ETo_setelagoas.csv') 
data_Patricia= pd.read_csv(path+'IC/2020_Ic/Dados/variaveis_setelagoas.csv')
data_Patricia["Eto"] = data_Patricia_Eto["Eto"]
if(os.path.exists(path+'IC/2020_Ic/Dados/data_PEto.csv')):
  print("JÁ EXISTE")
else:
  data_Patricia.to_csv(path+'IC/2020_Ic/Dados/data_PEto.csv')

atributeP= [ "Tmax","Tmean","I","UR","V","Tmin","J"]
resultadoP=get_x30(data_Patricia,atributeP,['Eto'])
if(os.path.exists(path+'IC/2020_Ic/Dados/Tabela30P_Lags.csv')):
  print("CSV patricia já existente !")
else:
  print("Exportando dados Patricia ...")
  resultadoP[0].to_csv(path+'IC/2020_Ic/Dados/Tabela30P_Lags.csv')
  print("Concluido !")
# resultados = arvore(data_Patricia)

# sns.boxplot (y = resultados['erro_rmse'] ); 
# plt.xlabel("Horizontes de previsão um dia", fontsize=14)  
# plt.ylabel("RMSE", fontsize=14)

# plt.show ()
arvore_parametros=[
                    [[['J']],[[0]],['Eto'],[1,2,3]],
                    [[['Tmax', 'I']],[[1],[1]],['Eto'],[1,2,3]],
                    [[['J', 'I']],[[0],[1]],['Eto'],[1,2,3]],
                    [[[ "Tmax","J","I"]],[[1],[0],[1]],['Eto'],[1,2,3]],
                    [[[ "Tmax","J","Tmean"]],[[1],[0],[1]],['Eto'],[1,2,3]],
                    [[["J"]],[[0]],['Eto'],[3]],
                    [[[ "J","Tmax"]],[[0],[3]],['Eto'],[3]],
                    [[["J","Tmax","I"]],[[0],[3],[3]],['Eto'],[3]],
                    [[["Tmax"]],[[3]],['Eto'],[3]],
                    [[["J"]],[[0]],['Eto'],[7]],
                    [[[ "J","Tmax"]],[[0],[7]],['Eto'],[7]],
                    [[["J","Tmax","I"]],[[0],[7],[7]],['Eto'],[7]],
                    [[["Tmax"]],[[7]],['Eto'],[7]],
                    [[["J"]],[[0]],['Eto'],[10]],
                    [[[ "J","Tmax"]],[[0],[10]],['Eto'],[10]],
                    [[[ "J","Tmax","I"]],[[0],[10],[10]],['Eto'],[10]],
                    [[["Tmax"]],[[10]],['Eto'],[10]]
                   ]
                   

print("\n Para o dataframe Patricia em dias")

lista2=['Tmax', 'Tmin', 'I', 'Tmean', 'UR', 'V', 'J']



# for x in [1,3,7,10]:
#   lista_Filtrada=column_Filter(resultadoP[0],x)
#   ranking_data_Patricia = resource_ranking(data_Patricia,lista_Filtrada,"Eto")
#   lista_Formatada1=list_Format(ranking_data_Patricia[0],10,0)
#   lista_Formatada2=list_Format(ranking_data_Patricia[1],10,0)
#   print('\n -----------Para----' + str(x) + '----dias --------\n')
#   print("lista_Formatada_1 -------> ",lista_Formatada1)
#   print("lista_Formatada_2 -------> ",lista_Formatada2)
#   print('\n ------------------------------------------\n')
  
  
  # parametrosSearchA=gridSearchAs(data_Patricia,[lista_Formatada1,lista_Formatada2],"Eto")
  # print(parametrosSearchA,"hiperparâmetro_da_Arvore")
  # parametrosSearchA=gridSearchFs(data_Patricia,[lista_Formatada1,lista_Formatada2],"Eto")
  # print(parametrosSearchA,"hiperparâmetro_da_Arvore")
  # parametrosSearchA=gridSearchXgbs(data_Patricia,[lista_Formatada1,lista_Formatada2],"Eto")
  # print(parametrosSearchA[0],"hiperparâmetro_da_Xgb")
  
# lista_Formatada_1 ------->  [['Tmean', 'Tmax'], [[1, 2, 3, 4], [1]], ['Eto'], [1, 2, 3, 4, 5]]
# lista_Formatada_2 ------->  [['Tmax', 'Tmean', 'I', 'UR', 'J'], [[1], [2], [1], [1], [22, 26]], ['Eto'], [1, 2, 3, 4]]
  
for x in [10]:
  lista_Filtrada=column_Filter(resultadoP[0],x)
  ranking_data_Patricia = resource_ranking(data_Patricia,lista_Filtrada,"Eto")
  lista_Formatada1=list_Format(ranking_data_Patricia[0],10,0)
  lista_Formatada2=list_Format(ranking_data_Patricia[1],10,0)

  print("lista_Formatada_1 -------> ",lista_Formatada1)
  print("lista_Formatada_2 -------> ",lista_Formatada2)
  print('\n -----------Para----' + str(x) + '----dias -------Modelo-\n')
  
  # arvores(data_Patricia,[lista_Formatada1,lista_Formatada2],"Eto")
#                      lista                      lista_lags    Eto         lags_eto  rmse
# 0            [Tmean, Tmax]             [[1, 2, 3, 4], [1]]  [Eto]  [1, 2, 3, 4, 5]  1.12
# 1  [Tmax, Tmean, I, UR, J]  [[1], [2], [1], [1], [22, 26]]  [Eto]     [1, 2, 3, 4]  1.08
  florestasAleatorias(data_Patricia,[lista_Formatada1,lista_Formatada2],"Eto")
# #                        lista                      lista_lags    Eto         lags_eto  rmse
# # 0            [Tmean, Tmax]             [[1, 2, 3, 4], [1]]  [Eto]  [1, 2, 3, 4, 5]  0.82
# # 1  [Tmax, Tmean, I, UR, J]  [[1], [2], [1], [1], [22, 26]]  [Eto]     [1, 2, 3, 4]  0.79
  # xgbs(data_Patricia,[lista_Formatada1,lista_Formatada2],"Eto")
#                      lista                      lista_lags    Eto         lags_eto    rmse
# 0            [Tmean, Tmax]             [[1, 2, 3, 4], [1]]  [Eto]  [1, 2, 3, 4, 5]  0.88
# 1  [Tmax, Tmean, I, UR, J]  [[1], [2], [1], [1], [22, 26]]  [Eto]     [1, 2, 3, 4]  0.87




# print("\n Para o dataframe larissa em horas")

# data_larissa = pd.read_csv(path+'IC/2020_Ic/Dados/Tabela_variaveis_larissa_original.csv') 

# lista3 =['vento','radiacao','temp_max','temp_min','umi_max','umi_min','umi_rel','press_atm']

# tab_la24h=tab_to_tab_lags(data_larissa,lista3,24)

# print(tab_la24h[0])

# lista_Filtrada2=column_Filter(tab_la24h[0],3)

# ranking_data_larissa = resource_ranking(data_larissa,lista_Filtrada2,"radiacao")

# lista_Formatada3=list_Format(ranking_data_larissa[0],0,0)
# lista_Formatada4=list_Format(ranking_data_larissa[1],0,0)

# print(lista_Formatada3)
# print(lista_Formatada4)

# arvores(data_larissa,[lista_Formatada3,lista_Formatada4],'radiacao')

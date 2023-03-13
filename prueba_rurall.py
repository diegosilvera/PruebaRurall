#Se leen los archivos de los fármacos contenidos en la base datos de entrenamiento y de test usando pandas
import pandas as pd
import re
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers import Dense


drugs_train = pd.read_csv('drugs_train.csv')
drugs_test = pd.read_csv('drugs_test.csv')

#Aquí comienza el preprocesamiento de los datos antes del entrenamiento
def add_one_if_not_number(s):
    if not s[0].isdigit():
        s = '1 ' + s
    return s
#La columna 'description' contiene instrucciones de cómo usar cada medicamento, en tipo de variable 'string'. 
#Se identifica que cada entrada de esa columna comienza por un número seguido de una palabra que contiene el substring '(s)'. En las entradas que no comienzan con números se añade al substring '1' al inicio.
#Este procedimiento ayuda a encontrar otros features en las bases de entrenamiento y de test.
drugs_train['description']= drugs_train['description'].apply(add_one_if_not_number)
drugs_test['description']= drugs_test['description'].apply(add_one_if_not_number)

#Se utiliza regex para extraer expresiones regulares de la columna 'description' para los datos de entrenamiento
#Esta expresión regular identifica dentro de un string patrones de números seguidos de un espacio y una palabra
regex = r'(\d+)\s+(\w+)'

#Con findall se encuentran todos los substrings de cada entrada que cumplen el patrón
result = drugs_train['description'].str.findall(regex)

#Se identifican todas las palabras únicas que aparecen de hacer la búsqueda por patrones regulares
unique_values = set()
for item in result:
    for value in item:
        unique_values.add(value[1])

# Se crea un dataframe con los valores únicos como columnas.
df = pd.DataFrame(columns=sorted(unique_values))

# Se llena el dataframe creado con los números asociados al patrón regular definido
for i, item in enumerate(result):
    for value in item:
        col = value[1]
        num = int(value[0])
        df.loc[i, col] = num

# Todos los NaN se remplazan con ceros. El cero quiere decir que para la entrada dada, la palabra dada por la columna no aparecía en la descripción.
df = df.fillna(0)

df.iloc[0]

#Se hace el mismo tratamiento por expresiones regulares para el set de test.
result = drugs_test['description'].str.findall(regex)
unique_values = set()
for item in result:
    for value in item:
        unique_values.add(value[1])


df_test = pd.DataFrame(columns=sorted(unique_values))


for i, item in enumerate(result):
    for value in item:
        col = value[1]
        num = int(value[0])
        df_test.loc[i, col] = num

# Replace NaN values with 0
df_test = df_test.fillna(0)

df_test.iloc[0]

#Dado que no todas las columnas nuevas extraídas a partir del feature engineering aparecen tanto en el set de entrenamiento como en el test, se busca hacer en este paso que las columnas de ambos sean iguales.
#Se añaden columnas que no aparecen en el set de train al set de test y viceversa, y se llenan con ceros.
for column in df.columns:
    if column not in df_test.columns:
        df_test[column]=np.zeros(len(df_test))
        print(column)
for column in df_test.columns:
    if column not in df.columns:
        df[column]=np.zeros(len(df))
        print(column)

#Algunas columnas de df solo difieren en su nombre por una 's' al final. Se agrupan de acuerdo a esto y se suman, para que no haya columnas duplicadas
grouped = df.groupby(df.columns.str.rstrip('s'), axis=1).sum()
grouped.columns

#Se crean nuevas columnas, una para cada columna original, con la etiqueta '_label' al final, de tal manera que sus valores son cero si en la palabra no aparece en la descripción de ese medicamento, y 1 si la palabra aparece
new_df = grouped.applymap(lambda x: 0 if x == 0 else 1).add_suffix('_label')
grouped=grouped.add_suffix('_count')

#Con las nuevas columnas creadas ahora tenemos el doble de columnas a partir del feature engineering de la columna 'description'
grouped=pd.concat([grouped,new_df],axis=1)
grouped

#La base de datos de entrenamientonueva ahora tiene varias columnas a partir del feature engineering de 'description'
drugs_train = pd.concat([drugs_train, grouped], axis=1)



df_test.columns

df.columns



#Se hace el mismo procesamiento de columnas para test.
grouped_test = df_test.groupby(df_test.columns.str.rstrip('s'), axis=1).sum()
grouped_test.columns

new_df_test = grouped_test.applymap(lambda x: 0 if x == 0 else 1).add_suffix('_label')
grouped_test=grouped_test.add_suffix('_count')

grouped_test=pd.concat([grouped_test,new_df_test],axis=1)
grouped_test

drugs_test = pd.concat([drugs_test, grouped_test], axis=1)



#Se extraen variables dummies de las columnas de tipo object para que se vuelvan numéricas, para set de entrenamiento y de prueba.
count=0
for column in drugs_train.columns:
    print(type(drugs_train[column]),column)
    if column=='drug_id' or drugs_train[column].dtype!='object' or column=='description' :
        continue
    else:
        dummies = pd.get_dummies(drugs_train[column])
        dummies_test = pd.get_dummies(drugs_test[column])
        if count == 0:
            merged_info = pd.concat([drugs_train.drop(['drug_id','description'],axis='columns'), dummies], axis='columns')
            merged_info_test = pd.concat([drugs_test.drop(['drug_id','description'],axis='columns'), dummies_test], axis='columns')
            
            count = 1
        else:
            merged_info = pd.concat([merged_info, dummies], axis='columns')
            merged_info_test = pd.concat([merged_info_test, dummies_test], axis='columns')
        print(column)
        unique_values = drugs_train[column].unique().tolist()
        merged_info.drop([column,unique_values[-1]], axis='columns',inplace=True)
        try:
            merged_info_test.drop([column,unique_values[-1]], axis='columns',inplace=True)
        except:
            merged_info_test.drop([column],axis='columns',inplace=True)
            print('no se pudo')

merged_info.dtypes

#Se hace que los features para la base de entrenamiento sean iguales que la de la base de prueba.
for column in merged_info.columns:
    if column not in merged_info_test.columns:
        merged_info_test[column]=np.zeros(len(merged_info_test))
        print(column)
for column in merged_info_test.columns:
    if column not in merged_info.columns:
        merged_info[column]=np.zeros(len(merged_info))
        print(column)

#Antes de seleccionar los features para entrenar, se separan las variables predictivas de las objetivo, X y Y, respectivamente, de los datos de entrenamiento. 

X = merged_info.drop(columns=['price'],axis=1)

y = merged_info["price"]
scaler = StandardScaler()
#X se escala a valores entre -1 y 1 para cada una de las variables predictivas, para que no haya ningún sesgo en el modelo
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled,columns=X.columns)

merged_info_test.dtypes

#Se escalan las variables predictivas para el set de prueba.
X_test = merged_info_test

scaler = StandardScaler()
#X se escala a valores entre -1 y 1 para cada una de las variables predictivas, para que no haya ningún sesgo en el modelo
X_scaled_test = scaler.fit_transform(X_test)
X_test = pd.DataFrame(X_scaled_test,columns=X_test.columns)

#Utilizando el módulo SelectFromModel y el algoritmo de regresión lineal Lasso, se seleccionan los 50 mejores features

#Se define el modelo Lasso con cross validation
clf = LassoCV(cv=5, random_state=0)

# Se seleccionan features usando SelectFromModel
sfm = SelectFromModel(clf,max_features=50)
sfm.fit(X, y)

X_selectedd = sfm.transform(X)

# Se imprimen los features seleccionados y sus indices
support = sfm.get_support()
selected_featuress = [i for i in range(len(support)) if support[i]]
print("Selected Features:", selected_featuress)



#Con el mismo modelo de regresión Lasso se entrena el modelo en los features seleccionados, para probar su desempeño. Se usan las métricas mean squared error, mean absolute error y r2 para medir diferentes formas

clf.fit(X[X.columns[selected_featuress]], y)

y_pred = clf.predict(X[X.columns[selected_featuress]])
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
#Se imprime el error cuadrático medio y el r2 para las variables de entrenamiento, con y sin cross validation
print('Mean Squared Error:', mse)
print('R-Squared:', r2)
scores = cross_val_score(clf, X[X.columns[selected_featuress]], y, cv=5, scoring='r2')
print("Average R-squared score from 5-fold cross-validation:", scores.mean())

#El desempeño con Lasso no es tan bueno, y se decide escoger una red neuronal profunda para predecir, después de haber probado con otros algoritmos como GradientBoosting, KNearestNeighbors, RandomForest, etc


# Se define cross validation para 10 splits, para validar el desempeño del modelo ante datos que no conoce
kf = KFold(n_splits=10, shuffle=True, random_state=10)


# Se inicializa una  lista para guardar los errores cuadráticos medios de cada iteración de cross validation, además de otras dos para guardar r2 y error medio absoluto
mse_scores = []
r2_scores = []
mae_scores = []
#Se define un modelo de red neuronal de 5 capas: una de 128 neuronas, de 64, 32, 16 y 1. La última capa da la predicción
Xp=X[X.columns[selected_featuress]]
nn = Sequential()
nn.add(Dense(128, input_dim=Xp.shape[1], activation='relu'))
nn.add(Dense(64, activation='relu'))
nn.add(Dense(32, activation='relu'))
nn.add(Dense(16, activation='relu'))
nn.add(Dense(1))
#Se Utiliza mean squared error como función de pérdida, y un optimizador adam.
nn.compile(loss='mean_squared_error', optimizer='adam')

#Se hace cross validation para una red neuronal como la especificada antes y con hiperparámetros epochs=30 y batch_size=16. Esto dio el mejor desempeño obtenido
for train_index,test_index in kf.split(Xp):

    print(0)
    X_train, X_test_val = Xp.iloc[train_index], Xp.iloc[test_index]
    y_train, y_test_val = y[train_index], y[test_index]
    nn.fit(X_train, y_train, epochs=30, batch_size=16, verbose=0)
    y_pred = nn.predict(X_test_val)
    accuracy = nn.evaluate(X_train, y_train, verbose=0)
    print(accuracy)
    accuracy = nn.evaluate(X_test_val, y_test, verbose=0)
    print(accuracy,mean_squared_error(y_test, y_pred),mean_absolute_error(y_test, y_pred),r2_score(y_test, y_pred))
    mse_scores.append(mean_squared_error(y_test, y_pred))
    mae_scores.append(mean_absolute_error(y_test, y_pred))
    r2_scores.append(r2_score(y_test, y_pred))    
# Se calcula el error cuadrático medio promedio, la desviación estándar del error cuadrático medio, el r2 y el error absoluto medio 
mse_mean = np.mean(mse_scores)
mse_std = np.std(mse_scores)
avg_mae = np.mean(mae_scores)
avg_r2 = np.mean(r2_scores)
print("Mean Squared Error (Mean):", mse_mean)
print("Mean Squared Error (Standard Deviation):", mse_std)
print('Average MAE:', avg_mae)
print('Average R^2:', avg_r2)

nn.fit(Xp, y, epochs=30, batch_size=16, verbose=0)

y_pred_test = nn.predict(X_test[X.columns[selected_featuress]])
#Aquí se predicen los precios asociados a los medicamentos con la red neuronal, para los features escogidos

len(y_pred_test)

merged_info_test

y_pred_test = list(a[0] for a in y_pred_test)

#Se  da el resultado en un archivo csv, en el formato solicitado, para los datos de prueba.
submission=pd.DataFrame({'drug_id':drugs_test['drug_id'].to_list(),'price':y_pred_test})
submission.to_csv('submission.csv',index=False)
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
from collections import Counter
from tensorflow import keras

# Se abre el archivo
data = pd.read_csv('/home/rafa/Dropbox/Postgrados/MDS/deeplearning/'
                   'tareas_deeplearning/tarea1/creditcard.csv')

# Se revisa el tipo de datos
data.info()

# Se revisa cuántas filas y columnas tiene el dataset
data.shape

# Se muestra una descripción estadística de los datos
data.describe()

# Se revisan las columnas del dataset
data.columns

# Se revisa si es que hay datos nulos
data.isnull().sum()

# Se indican las variables independientes y la dependiente
X_data = data.iloc[:, 0:30]
y_data = data.iloc[:, -1]

# Standar Scales
standard_scaler = preprocessing.StandardScaler()
X_standard_scaled_df = standard_scaler.fit_transform(X_data)

# Se imprime el df escalado obtenido.
X_standard_scaled_df

# Orden y creando el df escalado
X_standard_scaled_df = pd.DataFrame(data=X_standard_scaled_df[:, :],
                                    columns=['Time', 'V1', 'V2', 'V3', 'V4',
                                             'V5', 'V6', 'V7', 'V8', 'V9',
                                             'V10', 'V11', 'V12', 'V13', 'V14',
                                             'V15', 'V16', 'V17', 'V18', 'V19',
                                             'V20', 'V21', 'V22', 'V23', 'V24',
                                             'V25', 'V26', 'V27', 'V28',
                                             'Amount'])

# PCA
pca = PCA(10)

pca_selected = pca.fit_transform(X_standard_scaled_df)

# Se comprueba que efectivamente haya reducida solo el numero de columnas
print(pca_selected.shape)

# Se crea al df
pca_selected_df = pd.DataFrame(data=pca_selected[:, :])

# Df dinal para trabajar
ready_data = pca_selected_df.join(y_data)

# Se revisa la cantidad de 1 y 0 para determinar el balance del dataset
# Notamos su desbalance
data.Class.value_counts()

# Casi el 100% de los datos están etiquetados como categoría 0 de no fraude
data.Class.value_counts('1')

# Gráficamente mostrando el desbalance
sns.countplot(x="Class", data=data)
plt.show()

# Df que contiene la clase 0
data_class_0 = ready_data[ready_data['Class'] == 0]

# Dimensión del dataset de clase 0
data_class_0.shape

# Df que contiene la clase 1
data_class_1 = ready_data[ready_data['Class'] == 1]

# Dimensión del dataset de clase 1
data_class_1.shape

# Debido a que que se nota que el dataset está desbalanceado, 
# me preocuparé de balancearlo

# Columnas independientes
X_0 = data_class_0.iloc[:, 0:-1]

# Columna dependiente u objetivo
y_0 = data_class_0.iloc[:, -1] 

# Columnas independientes
X_1 = data_class_1.iloc[:, 0:-1]

# Columna dependiente u objetivo
y_1 = data_class_1.iloc[:, -1]

# Haciendo un split de los datos dejando un 20% para testeo y un 80% para
# entraniento del modelo
X_train_0, X_test_0, y_train_0, y_test_0 = train_test_split(X_0, y_0,
                                                            test_size=0.20,
                                                            random_state=42)
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_1, y_1,
                                                            test_size=0.20,
                                                            random_state=42)

# Concateno los conjuntos de entramiento y testeo, tanto para las variables
# dependientes como independientes
X_train = pd.concat([X_train_0, X_train_1])
y_train = pd.concat([y_train_0, y_train_1])
X_test = pd.concat([X_test_0, X_test_1])
y_test = pd.concat([y_test_0, y_test_1])

# Reviso que la dimensión sea la correcta
X_train.shape

# Reviso que la dimensión sea la correcta
y_train.shape

# Reviso que la dimensión sea la correcta
X_test.shape

# Reviso que la dimensión sea la correcta
y_test.shape

# SMOTE
print('Dimensión original de dataset %s' % Counter(y_train))

# Aplicando SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)
print('Dimensión luego de aplicado el SMOTE %s' % Counter(y_res))

# RL
# Se llama a la función de sklear con el solver lbgfs
logisticRegr = LogisticRegression(solver='lbfgs', max_iter=150)

# Se hace el fit creando el modelo
logit_model = logisticRegr.fit(X_train, y_train)

# Se crea el objeto para guardar los resultados pronosticados
# para el 20% seleccionado para testeo
logit_predict = logisticRegr.predict(X_test)

# Matriz de confusión
# Noto que son pocos los errores que comete FN=52 y FP=5
confusion_matrix(y_test, logit_predict)

# Calculo el recall
recall_score(y_test, logit_predict)

print(classification_report(y_test, logit_predict))

# Se llama a la función de sklearn con el solver lbgfs
logisticRegr2 = LogisticRegression(solver='newton-cg', max_iter=1000)

# Se hace el fit creando el modelo
logit_model = logisticRegr2.fit(X_train, y_train)

# Se crea el objeto para guardar los resultados pronosticados
# para el 20% seleccionado para testeo
logit_predict2 = logisticRegr2.predict(X_test)

# Matriz de confusión
# Noto que son pocos los errores que comete FN=52 y FP=5
confusion_matrix(y_test, logit_predict2)

# Calculo el accuracy
recall_score(y_test, logit_predict2)

print(classification_report(y_test, logit_predict2))

# Se inicializa la red neuronal
classifier = keras.Sequential()

# Agregando la capa input y la primera capa oculta
classifier.add(
    keras.layers.Dense(units=10, kernel_initializer='uniform',
                       activation='relu', input_dim=10))

# Agregando la capa de salida
classifier.add(
    keras.layers.Dense(units=1, kernel_initializer='uniform',
                       activation='sigmoid'))

# Compilando la red neuronal y agregando recall como medida a maximizar
classifier.compile(optimizer='adam', loss='binary_crossentropy',
                   metrics=['Recall'])

# Reusmen del clasificador
classifier.summary()

# Haciendo fit a la red neuronal al conjunto de entraniento
model = classifier.fit(X_train.values, y_train.values,
                       batch_size=128, epochs=5)

# Prediciendo los resultados en el conjunto de testeo
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
score = classifier.evaluate(X_test, y_test)
score

print(classification_report(y_test, y_pred))

# Genero una predicción de no fraude (clase mayoritaria)
ns_probs = [0 for _ in range(len(y_test))]

# Predicciones
lr_probs = classifier.predict(X_test)

# Mantengo las probabilidades para la salida positiva solamente
lr_probs = lr_probs[:, 0]

# Calculo los score
ns_auc = roc_auc_score(y_test, ns_probs)
lr_auc = roc_auc_score(y_test, lr_probs)

# Resumen
print('No Fraude: ROC AUC=%.3f' % (ns_auc))
print('ROC AUC: ROC AUC=%.3f' % (lr_auc))

# Iniciando
classifier2 = keras.Sequential()

classifier2.add(keras.layers.Dense(units=100, kernel_initializer='uniform',
                                   activation='relu', input_dim=10))

classifier2.add(keras.layers.Dense(units=60, kernel_initializer='uniform',
                                   activation='relu', input_dim=100))

classifier2.add(keras.layers.Dense(units=40, kernel_initializer='uniform',
                                   activation='relu', input_dim=60))

classifier2.add(keras.layers.Dense(units=25, kernel_initializer='uniform',
                                   activation='relu', input_dim=40))

classifier2.add(keras.layers.Dense(units=15, kernel_initializer='uniform',
                                   activation='relu', input_dim=25))

classifier2.add(keras.layers.Dense(units=5, kernel_initializer='uniform',
                                   activation='relu', input_dim=15))

classifier2.add(keras.layers.Dense(units=3, kernel_initializer='uniform',
                                   activation='relu', input_dim=5))

classifier2.add(keras.layers.Dense(units=2, kernel_initializer='uniform',
                                   activation='relu', input_dim=3))

classifier2.add(keras.layers.Dense(units=1, kernel_initializer='uniform',
                                   activation='sigmoid'))

classifier2.compile(optimizer='adam', loss='binary_crossentropy',
                    metrics=['Recall'])

# Resumen
classifier2.summary()

model2 = classifier2.fit(X_train.values, y_train.values,
                         batch_size=128, epochs=5)

# Prediciendo
y_pred2 = classifier2.predict(X_test)
y_pred2 = (y_pred2 > 0.5)
score2 = classifier2.evaluate(X_test, y_test)
score2

# Excelente Recall comparado con los dos modelos anteriores
print(classification_report(y_test, y_pred2))

# Genero una predicción de no fraude (clase mayoritaria)
ns_probs = [0 for _ in range(len(y_test))]

# Predicciones
lr_probs = classifier2.predict(X_test)

# Mantengo las probabilidades para la salida positiva solamente
lr_probs = lr_probs[:, 0]

# Calculo los score
ns_auc = roc_auc_score(y_test, ns_probs)
lr_auc = roc_auc_score(y_test, lr_probs)

# Resumen
print('No Fraude: ROC AUC=%.3f' % (ns_auc))
print('ROC AUC: ROC AUC=%.3f' % (lr_auc))

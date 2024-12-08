import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. CARGO LOS DATASETS
train_path = '<ruta>/train.csv'
test_path = '<ruta>/data/test.csv'
predicciones_path = '<ruta>/predicciones.csv'

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)
df_predicciones = pd.read_csv(predicciones_path)

# ----------------------------------------------------------------------------------------- #

# 2. LIMPIEZA Y PREPARACION DE DATOS
# Elimino las columnas que no aportan nada al futuro entrenamiento del modelo
columns_to_drop = ["Name", "Ticket", "Cabin"]
df_train = df_train.drop(columns=columns_to_drop)
df_test = df_test.drop(columns=columns_to_drop + ["Unnamed: 0"])

# Completo los valores nulos del atributo 'Age' con la media
df_train["Age"] = df_train["Age"].fillna(df_train["Age"].mean())
df_test["Age"] = df_test["Age"].fillna(df_train["Age"].mean())

# Completo los valores nulos del atributo 'Embarked' con la moda
most_frequent_embarked = df_train["Embarked"].mode()[0]
df_train["Embarked"] = df_train["Embarked"].fillna(most_frequent_embarked)

# Completo los valores nulos del atributo 'Fare' en test con la media del train
df_test["Fare"] = df_test["Fare"].fillna(df_train["Fare"].mean())

# ----------------------------------------------------------------------------------------- #

# 3. TRANSFORMACION DA DATOS
# Creo una nueva variable llamada 'Familiares'
df_train["Familiares"] = df_train["SibSp"] + df_train["Parch"]
df_test["Familiares"] = df_test["SibSp"] + df_test["Parch"]

# One-hot encoding para variables categóricas, es decir, para el atributo Sex, lo subdivido en dos atributos, 
# uno para representar el genero masculino y otro para el femenino, con valores de tipo booloeanos
df_train = pd.get_dummies(df_train, columns=["Sex", "Embarked"])
df_test = pd.get_dummies(df_test, columns=["Sex", "Embarked"])

# ----------------------------------------------------------------------------------------- #

# 4. ENTRENAMIENTO DEL MODELO
# Divido en características (X) y variable objetivo (y)
X_train = df_train.drop(columns=["Survived", "PassengerId"])
y_train = df_train["Survived"]

X_test = df_test.drop(columns=["PassengerId"])

# Creo el modelo
model = RandomForestClassifier(random_state=0)

# Entreno el modelo
model.fit(X_train, y_train)

# ----------------------------------------------------------------------------------------- #

# 5. PREDICCIONES
# Realizar predicciones en el dataset de prueba
predictions = model.predict(X_test)

# Crear el dataset de predicciones
output = pd.DataFrame({'PassengerId': df_test["PassengerId"], 'Survived': predictions})
output.to_csv('predicciones.csv', index=False)

print("Predicciones guardadas en 'predicciones.csv'")

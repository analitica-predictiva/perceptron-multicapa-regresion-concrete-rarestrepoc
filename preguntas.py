"""
Pronostico de la resistencia del concreto usando redes neuronales
-----------------------------------------------------------------------------------------

La descripción del problema está disponible en:

https://jdvelasq.github.io/courses/notebooks/sklearn_supervised_10_neural_networks/1-02_pronostico_de_la_resistencia_del_concreto.html

"""

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn import datasets
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error



def pregunta_01():
    """
    Carga y separación de los datos en `X` `y`
    """
    # Lea el archivo `concrete.csv` y asignelo al DataFrame `df`
    df = pd.read_csv("concrete.csv")

    # Asigne la columna `strength` a la variable `y`.
    y = df["strength"]

    # Asigne una copia del dataframe `df` a la variable `X`.
    X = df.copy()

    # Remueva la columna `strength` del DataFrame `X`.
    X.drop(["strength"], axis=1, inplace=True)

    # Retorne `X` y `y`
    return X, y



def pregunta_02():
    """
    Preparación del dataset.
    """

    # Cargue los datos de ejemplo y asigne los resultados a `X` y `y`.
    X, y = pregunta_01()

    # Divida los datos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=12453,
    )

    # Retorne `X_train`, `X_test`, `y_train` y `y_test`
    return X_train, X_test, y_train, y_test


def pregunta_03():
    """
    Construcción del pipeline
    """

    # Cree un pipeline
    pipeline = Pipeline(
        steps=[
            (
                "minmaxscaler",
                MinMaxScaler(),  
            ),
            (
                "mlpregressor",
                MLPRegressor(),  
            ),
        ],
    )

    # Retorne el pipeline
    return pipeline


def pregunta_04():
    """
    Creación de la malla de búsqueda
    """

    param_grid = {
        "mlpregressor__hidden_layer_sizes": [(i,) for i in range(1, 9)],
        "mlpregressor__activation": ["relu"],
        "mlpregressor__learning_rate": ["adaptive"],
        "mlpregressor__momentum": [0.6, 0.7, 0.8],
        "mlpregressor__learning_rate_init": [0.01, 0.05, 0.1],
        "mlpregressor__max_iter": [10000],
        "mlpregressor__early_stopping": [True],
    }

    estimator = pregunta_03()

    gridsearchcv = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=5,
        scoring="r2"
    )

    return gridsearchcv


def pregunta_05():
    """
    Evalúe el modelo obtenido.
    """

    # Cargue las variables
    X_train, X_test, y_train, y_test = pregunta_02()

    # Obtenga el objeto GridSearchCV
    estimator = pregunta_04()

    # Entrene el estimador
    estimator.fit(X_train, y_train)

    # Pronostique para las muestras de entrenamiento y validación
    y_train_pred = estimator.predict(X_train)
    y_test_pred = estimator.predict(X_test)

    # Calcule el error cuadrático medio
    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)

    # Retorne el mse de entrenamiento y prueba
    return mse_train, mse_test

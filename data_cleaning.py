import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objs as go


import datadotworld as ddw
import time

def process_incd_and_mort_data(incd_filepath, mort_filepath):
    """
    Procesa los archivos de incidencia y mortalidad para limpiarlos y estandarizar su formato.

    La función lee dos archivos CSV, maneja diferentes codificaciones para asegurar la correcta lectura,
    renombra las columnas para una mejor claridad, y estandariza los códigos FIPS. Se eliminan columnas no necesarias
    y se renombran las restantes para consistencia.

    Args:
        incd_filepath (str): Ruta del archivo CSV para los datos de incidencia.
        mort_filepath (str): Ruta del archivo CSV para los datos de mortalidad.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Dos DataFrames, el primero contiene los datos de incidencia
                                           limpios y el segundo los datos de mortalidad limpios.

    """
    try:
        incddf = pd.read_csv(incd_filepath, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            incddf = pd.read_csv(incd_filepath, encoding='latin1')
        except UnicodeDecodeError:
            incddf = pd.read_csv(incd_filepath, encoding='ISO-8859-1')

    # Leer mort.csv
    mortdf = pd.read_csv(mort_filepath)

    # Limpieza y procesamiento
    incddf.rename(columns={' FIPS': 'FIPS'}, inplace=True)
    mortdf = mortdf[mortdf.FIPS.notnull()]
    incddf = incddf[incddf.FIPS.notnull()]

    mortdf['FIPS'] = mortdf.FIPS.apply(lambda x: str(int(x)))\
                                .astype(np.object_)\
                                .str.pad(5, 'left', '0')
    incddf['FIPS'] = incddf.FIPS.apply(lambda x: str(int(x)))\
                                .astype(np.object_)\
                                .str.pad(5, 'left', '0')

    incddf.drop(incddf.columns[[0,3,4,7,8,9]].values, axis=1, inplace=True)
    mortdf.drop(mortdf.columns[[0,2,4,5,7,8,9,10]], axis=1, inplace=True)

    incddf.rename(columns={incddf.columns[1]:'Incidence_Rate',
                           incddf.columns[2]:'Avg_Ann_Incidence'}, inplace=True)
    mortdf.rename(columns={mortdf.columns[1]:'Mortality_Rate',
                           mortdf.columns[2]:'Avg_Ann_Deaths'}, inplace=True)

    return incddf, mortdf




    """
    Obtiene los nombres de tablas estatales de un conjunto de datos de pobreza de EE. UU.

    Carga y filtra las tablas del conjunto de datos 'acs-2015-5-e-poverty' de Data.world. 
    Incluye solo las tablas con nombres de dos letras (códigos estatales) y excluye 'pr'.

    Returns:
        list: Lista de nombres de tablas filtradas.
    """
def filter_tables():
  
  pov = ddw.load_dataset('uscensusbureau/acs-2015-5-e-poverty')

  tables = [i for i in pov.tables if len(i) == 2]

  tables.remove('pr')
  return tables

  

def load_and_process_proverty(tables):

    """
    Carga y procesa datos de pobreza por estado y condado de EE. UU.

    Para cada nombre de tabla proporcionado, realiza una consulta al conjunto de datos 'acs-2015-5-e-poverty' 
    de Data.world para obtener datos específicos de pobreza. Concatena los resultados de todas las tablas 
    en un único DataFrame.

    Args:
        tables (list): Lista de nombres de tablas (códigos estatales) para consulta.

    Returns:
        DataFrame: Datos consolidados de pobreza con columnas estandarizadas.
    """

    cols = '`State`, `StateFIPS`, `CountyFIPS`, `AreaName`, `B17001_002`, `B17001_003`,'\
         '`B17001_017`'
    for i, state in enumerate(tables):
      if i == 0:
        povdf = ddw.query('uscensusbureau/acs-2015-5-e-poverty',
                  '''SELECT %s FROM `AK`
                     WHERE SummaryLevel=50''' % cols).dataframe
      else:
        df = ddw.query('uscensusbureau/acs-2015-5-e-poverty',
                       '''SELECT %s FROM `%s`
                          WHERE SummaryLevel=50''' % (cols, state.upper())).dataframe

        povdf = pd.concat([povdf, df], ignore_index=True)

    povdf['StateFIPS'] = povdf.StateFIPS.astype(np.object_)\
                                    .apply(lambda x: str(x))\
                                    .str.pad(2, 'left', '0')
    povdf['CountyFIPS'] = povdf.CountyFIPS.astype(np.object_)\
                                      .apply(lambda x: str(x))\
                                      .str.pad(3, 'left', '0')
    povdf.rename(columns={'B17001_002':'All_Poverty', 'B17001_003':'M_Poverty', 'B17001_017':'F_Poverty'},
             inplace=True)
    return povdf


def load_income_data(tables):
    """
    Carga y procesa datos de ingresos medios por estado y condado de EE. UU.

    Realiza consultas al conjunto de datos 'acs-2015-5-e-income' en Data.world para cada estado especificado.
    Combina los datos de todos los estados en un DataFrame, estandarizando y renombrando las columnas 
    relacionadas con los ingresos medios de diferentes grupos demográficos.

    Args:
        tables (list): Lista de códigos estatales para consulta.

    Returns:
        DataFrame: Datos consolidados de ingresos medios con columnas renombradas y estandarizadas.
    """
    
    cols = '`StateFIPS`, `CountyFIPS`,'\
       '`B19013_001`, `B19013A_001`, `B19013B_001`, `B19013C_001`, `B19013D_001`,'\
       '`B19013I_001`'

    for i, state in enumerate(tables):
        if i == 0:
            incomedf = ddw.query('uscensusbureau/acs-2015-5-e-income',
                  '''SELECT %s FROM `AK`
                     WHERE SummaryLevel=50''' % cols).dataframe
        else:
            df = ddw.query('uscensusbureau/acs-2015-5-e-income',
                       '''SELECT %s FROM `%s`
                          WHERE SummaryLevel=50''' % (cols, state.upper())).dataframe
            incomedf = pd.concat([incomedf, df], ignore_index=True)
    incomedf['StateFIPS'] = incomedf.StateFIPS.astype(np.object_)\
                                .apply(lambda x: str(x))\
                                .str.pad(2, 'left', '0')
    incomedf['CountyFIPS'] = incomedf.CountyFIPS.astype(np.object_)\
                                 .apply(lambda x: str(x))\
                                 .str.pad(3, 'left', '0')
    incomedf.rename(columns={'B19013_001':'Med_Income', 'B19013A_001':'Med_Income_White',
                         'B19013B_001':'Med_Income_Black', 'B19013C_001':'Med_Income_Nat_Am',
                         'B19013D_001':'Med_Income_Asian', 'B19013I_001':'Hispanic'}, inplace=True)
    return incomedf




    """
    Carga y procesa datos de cobertura de seguro médico por estado y condado de EE. UU.

    Realiza consultas al conjunto de datos 'acs-2015-5-e-healthinsurance' en Data.world para estados especificados,
    enfocándose en los datos de cobertura de seguro médico para diferentes grupos de edad y género.
    Procesa y suma las categorías para obtener totales de cobertura y no cobertura por género.

    Args:
        tables (list): Lista de códigos estatales para consulta.

    Returns:
        DataFrame: Datos consolidados de cobertura de seguro médico con columnas para hombres, mujeres y totales.
    """

def load_and_process_health_insurance_data(tables):
    cols = '`StateFIPS`, `CountyFIPS`,'\
           '`B27001_004`, `B27001_005`, `B27001_007`, `B27001_008`,'\
           '`B27001_010`, `B27001_011`, `B27001_013`, `B27001_014`,'\
           '`B27001_016`, `B27001_017`, `B27001_019`, `B27001_020`,'\
           '`B27001_022`, `B27001_023`, `B27001_025`, `B27001_026`,'\
           '`B27001_028`, `B27001_029`, `B27001_032`, `B27001_033`,'\
           '`B27001_035`, `B27001_036`, `B27001_038`, `B27001_039`,'\
           '`B27001_041`, `B27001_042`, `B27001_044`, `B27001_045`,'\
           '`B27001_047`, `B27001_048`, `B27001_050`, `B27001_051`,'\
           '`B27001_053`, `B27001_054`, `B27001_056`, `B27001_057`'
    for i, state in enumerate(tables):
        if i == 0:
            hinsdf = ddw.query('uscensusbureau/acs-2015-5-e-healthinsurance',
                      '''SELECT %s FROM `AK`
                         WHERE SummaryLevel=50''' % cols).dataframe
        else:
            df = ddw.query('uscensusbureau/acs-2015-5-e-healthinsurance',
                           '''SELECT %s FROM `%s`
                              WHERE SummaryLevel=50''' % (cols, state.upper())).dataframe
            hinsdf = pd.concat([hinsdf, df], ignore_index=True)

    hinsdf['StateFIPS'] = hinsdf.StateFIPS.astype(np.object_)\
                                      .apply(lambda x: str(x))\
                                      .str.pad(2, 'left', '0')
    hinsdf['CountyFIPS'] = hinsdf.CountyFIPS.astype(np.object_)\
                                        .apply(lambda x: str(x))\
                                        .str.pad(3, 'left', '0')


    males = ['`B27001_004`', '`B27001_005`', '`B27001_007`', '`B27001_008`',
               '`B27001_010`', '`B27001_011`', '`B27001_013`', '`B27001_014`',
               '`B27001_016`', '`B27001_017`', '`B27001_019`', '`B27001_020`',
               '`B27001_022`', '`B27001_023`', '`B27001_025`', '`B27001_026`',
               '`B27001_028`', '`B27001_029`']

    females = ['`B27001_032`', '`B27001_033`', '`B27001_035`', '`B27001_036`',
               '`B27001_038`', '`B27001_039`', '`B27001_041`', '`B27001_042`',
               '`B27001_044`', '`B27001_045`', '`B27001_047`', '`B27001_048`',
               '`B27001_050`', '`B27001_051`', '`B27001_053`', '`B27001_054`',
               '`B27001_056`', '`B27001_057`']

    males_with = []
    males_without = []
    females_with = []
    females_without = []

    for i, j in enumerate(males):
        if i % 2 == 0:
            males_with.append(j.replace('`', ''))
        else:
            males_without.append(j.replace('`', ''))
    for i, j in enumerate(females):
        if i % 2 == 0:
            females_with.append(j.replace('`', ''))
        else:
            females_without.append(j.replace('`', ''))

    clist = [males_with, males_without, females_with, females_without]
    newcols = ['M_With', 'M_Without', 'F_With', 'F_Without']
    for col in newcols:
        hinsdf[col] = 0


    for i in males_with:
        hinsdf['M_With'] += hinsdf[i]
    for i in males_without:
        hinsdf['M_Without'] += hinsdf[i]
    for i in females_with:
        hinsdf['F_With'] += hinsdf[i]
    for i in females_without:
        hinsdf['F_Without'] += hinsdf[i]

    hinsdf['All_With'] = hinsdf.M_With + hinsdf.F_With
    hinsdf['All_Without'] = hinsdf.M_Without + hinsdf.F_Without

# Remove all the individual age group variables
# but, save them as a df called hinsdf_extra (just in case)
    columns_to_remove = hinsdf.columns[hinsdf.columns.str.contains('B27001')].values
    hinsdf_extra = hinsdf.loc[:, columns_to_remove]
    hinsdf.drop(columns=columns_to_remove, axis=1, inplace=True)

    return hinsdf





    """
    Combina múltiples conjuntos de datos relacionados con salud y socioeconómicos en un único DataFrame.

    Fusiona conjuntos de datos sobre pobreza, ingresos, seguro médico, incidencia de enfermedades y tasas de mortalidad.
    Procesa y ajusta los datos para uniformidad, incluyendo la generación de códigos FIPS y la eliminación de columnas no necesarias.
    Incorpora estimaciones de población para cada FIPS y ajusta ciertos valores numéricos.

    Args:
        povdf (DataFrame): Datos de pobreza.
        incomedf (DataFrame): Datos de ingresos.
        hinsdf (DataFrame): Datos de cobertura de seguro médico.
        incddf (DataFrame): Datos de incidencia de enfermedades.
        mortdf (DataFrame): Datos de mortalidad.

    Returns:
        DataFrame: Un DataFrame consolidado con datos combinados para análisis más detallados.
    """
def create_combined_dataset(povdf, incomedf, hinsdf, incddf, mortdf):
    # Lista de DataFrames
    dfs = [povdf, incomedf, hinsdf, incddf, mortdf]

    for df in [povdf, incomedf, hinsdf]:
        df['FIPS'] = df.StateFIPS + df.CountyFIPS
        df.drop(['StateFIPS', 'CountyFIPS'], axis=1, inplace=True)

    for i, j in enumerate(dfs):
        if i == 0:
            fulldf = j.copy()
        else:
            fulldf = fulldf.merge(j, how='inner', on='FIPS')

    fulldf.drop(['Med_Income_White', 'Med_Income_Black', 'Med_Income_Nat_Am',
                 'Med_Income_Asian', 'Hispanic'], axis=1, inplace=True)

    populationdf = ddw.query('nrippner/us-population-estimates-2015',
                             '''SELECT `POPESTIMATE2015`, `STATE`, `COUNTY`
                                FROM `CO-EST2015-alldata`''').dataframe

    state = populationdf.STATE.apply(lambda x: str(x)).str.pad(2, 'left', '0')
    county = populationdf.COUNTY.apply(lambda x: str(x)).str.pad(3, 'left', '0')
    populationdf['FIPS'] = state + county
    fulldf = fulldf.merge(populationdf[['FIPS', 'POPESTIMATE2015']], on='FIPS', how='inner')

    # Adiciones a la función
    fulldf = fulldf[fulldf.Mortality_Rate != '*']
    fulldf['Med_Income'] = pd.to_numeric(fulldf.Med_Income)

    # Identificar valores que no se pueden convertir a numérico
    values = []
    for _, j in enumerate(fulldf.Incidence_Rate):
        try:
            pd.to_numeric(j)
        except:
            values.append(j)

    # Procesamiento adicional
    fulldf.rename(columns={'Recent Trend': 'RecentTrend'}, inplace=True)
    fulldf.replace({'RecentTrend': {'*': 'stable'}}, inplace=True)

    # Función para verificación booleana
    def f(x, term):
        return 1 if x == term else 0

    # Crear nuevas características
    fulldf['Rising'] = fulldf.RecentTrend.apply(lambda x: f(x, 'rising'))
    fulldf['Falling'] = fulldf.RecentTrend.apply(lambda x: f(x, 'falling'))

    return fulldf



    """
    Prepara los datos para análisis, separando las variables objetivo y predictoras.

    Convierte las tasas de mortalidad en valores numéricos como variable objetivo. Selecciona y procesa 
    variables predictoras relevantes, manejando valores faltantes y calculando tasas per cápita para ciertos campos.

    Args:
        fulldf (DataFrame): DataFrame con datos combinados para el análisis.

    Returns:
        tuple: Dos DataFrames, X con variables predictoras y y con la variable objetivo.
    """
def prepare_data_for_analysis(fulldf):
    # Preparar la variable objetivo
    y = pd.to_numeric(fulldf.Mortality_Rate).values

    # Preparar las variables predictoras
    X = fulldf.loc[:, ['All_Poverty', 'M_Poverty', 'F_Poverty', 'Med_Income',
                       'M_With', 'M_Without', 'F_With', 'F_Without', 'All_With',
                       'All_Without', 'Incidence_Rate', 'Falling', 'Rising',
                       'POPESTIMATE2015']]

    # Convertir y manejar valores faltantes en 'Incidence_Rate'
    X['Incidence_Rate'] = pd.to_numeric(X.Incidence_Rate, errors='coerce')
    X['Incidence_Rate'] = X.Incidence_Rate.fillna(X.Incidence_Rate.median())

    # Calcular tasas per cápita
    for col in ['All_Poverty', 'M_Poverty', 'F_Poverty', 'M_With',
                'M_Without', 'F_With', 'F_Without', 'All_With', 'All_Without']:
        X[col + "_PC"] = X[col] / X.POPESTIMATE2015 * 10**5

    # Eliminar columnas adicionales
    columns_to_drop = ['M_Poverty_PC', 'F_Poverty_PC', 'M_With_PC', 'F_With_PC',
                       'M_Without_PC', 'F_Without_PC']
    X.drop(columns_to_drop, axis=1, inplace=True)

    return X, y


    """
    Clase para limpiar y preparar datos para análisis estadístico.

    Esta clase integra múltiples conjuntos de datos relacionados con la salud y la demografía, realizando 
    operaciones de limpieza, combinación y preparación de datos.

    Attributes:
        incd_filepath (str): Ruta al archivo CSV con datos de incidencia.
        mort_filepath (str): Ruta al archivo CSV con datos de mortalidad.
        tables (list): Lista de nombres de tablas filtradas para análisis.
        povdf (DataFrame): Datos procesados de pobreza.
        incomedf (DataFrame): Datos procesados de ingresos.
        hinsdf (DataFrame): Datos procesados de seguro de salud.
        incddf (DataFrame): Datos procesados de incidencia.
        mortdf (DataFrame): Datos procesados de mortalidad.
        fulldf (DataFrame): Dataset combinado para análisis.
        X (DataFrame): Variables predictoras para el modelo.
        y (array): Variable objetivo para el modelo.

    Methods:
        run_all(): Ejecuta el proceso completo de limpieza y preparación de datos.
    """
class DataCleaning:
    def __init__(self, incd_filepath, mort_filepath):
        self.incd_filepath = incd_filepath
        self.mort_filepath = mort_filepath
        self.tables = None
        self.povdf = None
        self.incomedf = None
        self.hinsdf = None
        self.incddf = None
        self.mortdf = None
        self.fulldf = None
        self.X = None
        self.y = None

    def run_all(self):
        # Ejecuta las funciones en orden
        self.incddf, self.mortdf = process_incd_and_mort_data(self.incd_filepath, self.mort_filepath)
        self.tables = filter_tables()
        self.povdf = load_and_process_proverty(self.tables)
        self.incomedf = load_income_data(self.tables)
        self.hinsdf = load_and_process_health_insurance_data(self.tables)
        self.fulldf = create_combined_dataset(self.povdf, self.incomedf, self.hinsdf, self.incddf, self.mortdf)
        self.X, self.y = prepare_data_for_analysis(self.fulldf)


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


class LinearRegressionModel:
    """
    Clase para construir y evaluar un modelo de regresión lineal.

    Esta clase facilita el entrenamiento y evaluación de un modelo de regresión lineal usando datos proporcionados.
    Permite dividir los datos, entrenar el modelo, hacer predicciones y evaluar el rendimiento del modelo.

    Attributes:
        X (DataFrame): Variables independientes del conjunto de datos.
        y (array): Variable dependiente del conjunto de datos.
        model (LinearRegression): Instancia del modelo de regresión lineal.
        X_train, X_test (DataFrame): Datos divididos para entrenamiento y prueba (predictores).
        y_train, y_test (array): Datos divididos para entrenamiento y prueba (objetivo).
        y_pred (array): Predicciones realizadas por el modelo.

    Methods:
        split_data(test_size, random_state): Divide los datos en conjuntos de entrenamiento y prueba.
        train_model(): Entrena el modelo de regresión lineal.
        make_predictions(): Realiza predicciones con el modelo entrenado.
        evaluate_model(): Calcula y devuelve el error cuadrático medio, R2 y RMSE del modelo.
        get_coefficients(): Retorna los coeficientes del modelo como un DataFrame.
    """
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.model = LinearRegression()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None

    def split_data(self, test_size=0.2, random_state=42):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )

    def train_model(self):
        self.model.fit(self.X_train, self.y_train)

    def make_predictions(self):
        self.y_pred = self.model.predict(self.X_test)

    def evaluate_model(self):
        mse = mean_squared_error(self.y_test, self.y_pred)
        r2 = r2_score(self.y_test, self.y_pred)
        rmse = np.sqrt(mse)
        return mse, r2, rmse

    def get_coefficients(self):
        coefficients = pd.DataFrame(self.model.coef_, self.X.columns, columns=['Coefficient'])
        return coefficients

   
    """
    Clase para visualizar los resultados de un modelo de regresión lineal.

    Esta clase permite crear gráficos que muestran la relación entre los valores reales y los ajustados por el modelo de regresión lineal. También incluye métricas de rendimiento del modelo.

    Attributes:
        X (DataFrame): Variables independientes del conjunto de datos.
        y (array): Variable dependiente del conjunto de datos.
        model (LinearRegression): Instancia del modelo de regresión lineal utilizado.
        mse (float): Error cuadrático medio del modelo.
        r2 (float): Coeficiente de determinación R2 del modelo.
        rmse (float): Raíz del error cuadrático medio del modelo.

    Methods:
        plot_regression(): Crea y devuelve un gráfico de dispersión con la línea de mejor ajuste.
    """
import plotly.graph_objs as go
from plotly.offline import plot
import numpy as np

"""
    Clase para generar y visualizar predicciones a partir de un modelo de regresión lineal.

    Esta clase facilita la generación de datos aleatorios basados en las características del modelo y la visualización de las predicciones resultantes.

    Attributes:
        model (LinearRegression): Instancia del modelo de regresión lineal.
        X_features (DataFrame): DataFrame de características basado en el cual se generan datos aleatorios.
        num_samples (int): Número de muestras aleatorias a generar.
        random_X (DataFrame): Datos aleatorios generados.
        predictions (array): Predicciones realizadas por el modelo en los datos aleatorios.

    Methods:
        generate_random_data(): Genera datos aleatorios basados en las características del modelo.
        make_predictions(): Realiza predicciones en los datos aleatorios generados.
        plot_predictions_line(): Crea y devuelve un gráfico de línea de las predicciones.
"""

class RegressionPlot:
    
    def __init__(self, X, y, model, mse, r2, rmse):
        self.X = X
        self.y = y
        self.model = model
        self.mse = mse
        self.r2 = r2
        self.rmse = rmse

    def plot_regression(self):
        # Calcular los valores ajustados y la línea de mejor ajuste
        fitted_values = self.model.predict(self.X)
        slope, intercept = np.polyfit(self.y, fitted_values, 1)
        line_x = np.linspace(min(self.y), max(self.y), 100)
        line_y = slope * line_x + intercept

        # Crear el gráfico de dispersión con nodos de dispersión fucsia
        scatter = go.Scatter(
            x=self.y,
            y=fitted_values,
            mode='markers',
            marker=dict(
                color='fuchsia',  # Color fucsia para los nodos
                opacity=0.7       # Opacidad ajustada para mejor visualización
            ),
            name='Data'
        )

        # Crear la línea de regresión con un color contrastante
        line = go.Scatter(
            x=line_x,
            y=line_y,
            mode='lines',
            line=dict(
                color='lime',   # Color lima para la línea, que contrasta bien con fucsia
                width=2,
                dash='dash'
            ),
            name='Fit'
        )

        # Añadir las trazas a la figura
        fig = go.Figure(data=[scatter, line])

        # Actualizar el diseño de la figura
        fig.update_layout(
            title=f'Actual Values vs Fitted Values<br>MSE: {self.mse:.4f}, R2: {self.r2:.4f}, RMSE: {self.rmse:.4f}',
            xaxis_title='Actual Values',
            yaxis_title='Fitted Values',
            yaxis=dict(range=[0, max(self.y) + 10]),
            template='none',
            plot_bgcolor='rgb(45,45,45)',  # Fondo gris para la gráfica
            paper_bgcolor='rgb(45,45,45)', # Fondo gris para el área alrededor de la gráfica
            font_color='white',   # Color blanco para el texto, para mejor contraste
            height=600,           # Ajustado para un tamaño más grande
            width=1000            # Ajustado para un tamaño más grande
        )

        # Ajustar colores del texto y fondo de los ejes
        fig.update_xaxes(
            showgrid=True, gridwidth=1, gridcolor='lightgrey',
            zerolinecolor='grey', linecolor='grey'  # Ajustes de color para el contorno y líneas cero
        )
        fig.update_yaxes(
            showgrid=True, gridwidth=1, gridcolor='lightgrey',
            zerolinecolor='grey', linecolor='grey'  # Ajustes de color para el contorno y líneas cero
        )

        # Retornar la gráfica como un div HTML para su uso en Flask
        plot_div = plot(fig, output_type='div', include_plotlyjs=False)
        return plot_div





    

import json
import plotly.graph_objs as go
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

class PredictionModel:
    def __init__(self, model, X_features, num_samples=100):
        self.model = model
        self.X_features = X_features
        self.num_samples = num_samples

    def generate_random_data(self):
        random_data = {}
        for column in self.X_features.columns:
            mean = self.X_features[column].mean()
            std = self.X_features[column].std()
            random_data[column] = np.random.normal(mean, std, self.num_samples)
        self.random_X = pd.DataFrame(random_data)

    def make_predictions(self):
        self.predictions = self.model.predict(self.random_X)

    def plot_predictions_line(self):
        mean_prediction = np.mean(self.predictions)

        line_chart = go.Scatter(
            x=np.arange(self.num_samples),
            y=self.predictions,
            mode='lines+markers',
            marker=dict(color='#FF00FF', size=8),
            line=dict(color='lightblue', width=3),
            name='Predictions'
        )

        mean_line = go.Scatter(
            x=np.arange(self.num_samples),
            y=[mean_prediction] * self.num_samples,
            mode='lines',
            line=dict(color='pink', width=3, dash='dash'),
            name='Average Predictions'
        )

        fig = go.Figure(data=[line_chart, mean_line])

        fig.update_layout(
            title='Predictions for Random Data (Line Chart)',
            title_font_color='white',
            xaxis_title='Sample',
            yaxis_title='Prediction',
            plot_bgcolor='rgb(45,45,45)',
            paper_bgcolor='rgb(45,45,45)',
            font=dict(color='white'),
            height=600,
            width=1000,
            legend=dict(x=0.01, y=0.99, bgcolor='rgba(0,0,0,0)', bordercolor='red')
        )

        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='grey')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='grey')

        # Convertir la figura a JSON y devolver el diccionario
        return json.loads(fig.to_json())




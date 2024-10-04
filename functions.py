import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import scipy.stats as stats
import numpy as np
from scipy.stats import mannwhitneyu

def drop_info_null(df, col):
    """
    Elimina las filas con valores nulos en la columna especificada (col) de un DataFrame (df),
    imprime el número de filas eliminadas.

    Parameters
    ----------
    df : DataFrame
        El DataFrame desde el que se eliminarán las filas con valores nulos
    col : str
        El nombre de la columna que contiene los valores nulos a eliminar
    """

    # Cuenta el número total de nulos en la columna especificada
    total_nulls = df[col].isna().sum()

    # Elimina las filas con nulos en la columna especificada, modificando el DataFrame original
    df.dropna(subset=[col], inplace=True)

    # Imprime el número de filas eliminadas
    print(f"Había {total_nulls} filas con valores nulos que se han eliminado de {col}")
def replace_values(df, col, replace_dict):
    """
    Reemplaza los valores de una columna (col) de un DataFrame (df) según un diccionario (replace_dict).
    
    Parameters
    ----------
    df : DataFrame
        El DataFrame cuya columna se va a reemplazar
    col : str
        El nombre de la columna que se va a reemplazar
    replace_dict : dict
        El diccionario que contiene los valores a reemplazar como clave y el valor a reemplazar como valor
        
    Returns
    -------
    None
    """
    if col not in df.columns:
        print(f"Error: La columna '{col}' no está presente en el DataFrame.")
        return
    df[col] = df[col].replace(replace_dict)
    print(f"Se han reemplazado los valores de la columna '{col}' según el diccionario proporcionado.")
def categorize_age_groups(df,col, new_col, start, end, interval):

    # Definir los bins para los grupos de edad
    bins = list(range(start, end + 1, interval)) + [float('inf')]  # Agrega 'inf' 

    # Crear las etiquetas para los grupos de edad
    labels = [f"{i}-{i + interval - 1}" for i in bins[:-2]] + [f"{bins[-2]}+"]

    # Agrupar las edades en intervalos de 5 años
    df[new_col] = pd.cut(df[col], bins=bins, labels=labels, right=False)

    return df
def group_by_col_agg(df, first_cols, last_col, group_by_col):
    """
    :param df: DataFrame .
    :param first_cols: Lista de columnas que usarán la función 'first'.
    :param last_col: Columna que usará la función 'last'.
    :param group_by_col: Columna por la que se agrupa.
    :return: DataFrame con el análisis único.
    """
    
    # Crear el diccionario de agregación
    agg_dict = {last_col: 'last'}  # Columna que usará 'last'
    
    # Agregar columnas con 'first'
    agg_dict.update({col: 'first' for col in first_cols})
    
    # Agrupación y agregación
    df = df.groupby(group_by_col).agg(agg_dict).reset_index()
    
    return df
def plot_separate_metrics_by_group(df, group_col, metric_col, metric_label):
    metrics = df.groupby(group_col).agg({metric_col: ['mean', 'sum']}).reset_index()
    metrics.columns = [group_col, f'{metric_col}_mean', f'{metric_col}_sum']  # Renombrar columnas

    # Gráfico de Media
    plt.figure(figsize=(10, 6))
    plt.bar(metrics[group_col], metrics[f'{metric_col}_mean'], color='blue', label=f'Media de {metric_label}')
    plt.title(f'Media de {metric_label} por {group_col.capitalize()}', fontsize=14)
    plt.ylabel(f'Cantidad de {metric_label}', fontsize=12)
    plt.xlabel(group_col, fontsize=12)
    plt.xticks(rotation=45)  # Rotar etiquetas si es necesario
    plt.legend()
    plt.grid(True)
    plt.show()

    # Gráfico de Suma
    plt.figure(figsize=(10, 6))
    plt.bar(metrics[group_col], metrics[f'{metric_col}_sum'], color='orange', label=f'Total de {metric_label}')
    plt.title(f'Total de {metric_label} por {group_col.capitalize()}', fontsize=14)
    plt.ylabel(f'Cantidad de {metric_label}', fontsize=12)
    plt.xlabel(group_col, fontsize=12)
    plt.xticks(rotation=45)  # Rotar etiquetas si es necesario
    plt.legend()
    plt.grid(True)
    plt.show()
def plot_distribution(data, plot_type, x, title, xlabel, ylabel, rotation=None, **kwargs):


    """
    Función para generar gráficos de distribución.
    
    Parameters:
        data: DataFrame de pandas que contiene los datos.
        plot_type: Tipo de gráfico ('histplot' o 'countplot').
        x: Nombre de la columna en el eje x.
        title: Título del gráfico.
        xlabel: Etiqueta del eje x.
        ylabel: Etiqueta del eje y.
        rotation: Rotación de las etiquetas del eje x (opcional).
        kwargs: Argumentos adicionales para personalizar el gráfico.
    """
    plt.figure(figsize=(10, 6))  # Ajusta el tamaño de la figura
    if plot_type == 'histplot':
        sns.histplot(data[x], **kwargs)
    elif plot_type == 'countplot':
        sns.countplot(data=data, x=x, **kwargs)
        if rotation is not None:  # Aplica la rotación solo si se especifica
            plt.xticks(rotation=rotation)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
def analyze_contingency_table(dataframe, row_variable, column_variable):

    """
    Analiza la tabla de contingencia para las variables especificadas y realiza la prueba chi-cuadrado.
    
    Parameters:
    - dataframe: DataFrame de pandas que contiene los datos.
    - row_variable: Nombre de la columna para el eje de filas.
    - column_variable: Nombre de la columna para el eje de columnas.

    Returns:
    - chi2: estadístico chi-cuadrado
    - p: valor p
    - contingency_table: tabla de contingencia normalizada
    """
    # Crear tabla de contingencia normalizada
    contingency_table = pd.crosstab(dataframe[row_variable], dataframe[column_variable], normalize='columns')
    
    # Realizar la prueba chi-cuadrado
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    
    # Imprimir resultados
    print(f'Chi2: {chi2}, P-value: {p}')
    print("Tabla de Contingencia Normalizada:")
    print(contingency_table)
    
    return chi2, p, contingency_table



    """
    Realiza un análisis único de clientes agrupando por 'client_id'.

    :param df: DataFrame .
    :param first_cols: Lista de columnas que usarán la función 'first'.
    :param last_col: Columna que usará la función 'last'.
    :param group_by_col: Columna por la que se agrupa.
    :return: DataFrame con el análisis único.
    """
    
    # Crear el diccionario de agregación
    agg_dict = {last_col: 'last'}  # Columna que usará 'last'
    
    # Agregar columnas con 'first'
    agg_dict.update({col: 'first' for col in first_cols})
    
    # Agrupación y agregación
    df = df.groupby(group_by_col).agg(agg_dict).reset_index()
    
    return df
def unique_client_analysis(df, first_cols, last_col, group_by_col):


    """
    Realiza un análisis único de clientes agrupando por 'client_id'.

    :param df: DataFrame .
    :param first_cols: Lista de columnas que usarán la función 'first'.
    :param last_col: Columna que usará la función 'last'.
    :param group_by_col: Columna por la que se agrupa.
    :return: DataFrame con el análisis único.
    """
    
    # Crear el diccionario de agregación
    agg_dict = {last_col: 'last'}  # Columna que usará 'last'
    
    # Agregar columnas con 'first'
    agg_dict.update({col: 'first' for col in first_cols})
    
    # Agrupación y agregación
    df = df.groupby(group_by_col).agg(agg_dict).reset_index()
    
    return df
def classify_clients(df, bal_col, accts_col, high_percentile=0.85, mid_percentile=0.40, high_value=0, mid_value=1, low_value=2):
    """
    Clasifica a los clientes en categorías según los percentiles del balance y número de cuentas.
    
    Parameters:
    df: DataFrame -> El DataFrame con los datos de los clientes.
    bal_col: str -> Nombre de la columna del balance.
    accts_col: str -> Nombre de la columna del número de cuentas.
    high_percentile: float -> Percentil para clasificar en la categoría alta (por defecto 0.85).
    mid_percentile: float -> Percentil para clasificar en la categoría media (por defecto 0.40).
    high_value: any -> Valor para clasificar en la categoría alta (por defecto 0).
    mid_value: any -> Valor para clasificar en la categoría media (por defecto 1).
    low_value: any -> Valor para clasificar en la categoría baja (por defecto 2).
    
    Returns:
    Series -> Columna con la clasificación de cada cliente.
    """
    # Calcular los percentiles de balance y número de cuentas
    high_bal = df[bal_col].quantile(high_percentile)
    high_accts = df[accts_col].quantile(high_percentile)
    
    mid_bal = df[bal_col].quantile(mid_percentile)
    mid_accts = df[accts_col].quantile(mid_percentile)
    
    # Función interna para clasificar cada fila
    def classify_row(row):
        if row[bal_col] >= high_bal and row[accts_col] >= high_accts:
            return high_value  # Categoría alta
        elif row[bal_col] >= mid_bal and row[accts_col] >= mid_accts:
            return mid_value  # Categoría media
        else:
            return low_value  # Categoría baja
    
    # Aplicar la clasificación fila por fila
    return df.apply(classify_row, axis=1)


def salto_mayor_a_1(dataf):

    results = []
    # Verificar si el grupo tiene al menos dos elementos
    if len(dataf) < 2:
        return False 
    
    for i in range(1, len(dataf)):
        if abs(dataf['process_step'].iloc[i] - dataf['process_step'].iloc[i - 1]) > 1:
            d = {'client_id': dataf['client_id'].iloc[i],
                 'visitor_id': dataf['visitor_id'].iloc[i],
                 'visit_id': dataf['visit_id'].iloc[i],
                 'first_step': dataf['process_step'].iloc[i],
                 'sec_step': dataf['process_step'].iloc[i - 1]}
            results.append(d)  # Append the result to the list

    if results:
        return pd.DataFrame(results)
    return False


def process_data(df):
    # Ordenar por 'visit_id' y 'date_time' en orden descendente
    df = df.sort_values(['visit_id', 'date_time'], ascending=False)
    
    # Crear una columna para los errores de regresión
    df['regr_error'] = False
    
    # Reiniciar el índice
    df = df.reset_index(drop=True)
    
    # Agregar columnas con el siguiente paso de proceso y la siguiente visita
    df['next_process_step'] = df['process_step'].shift(-1)
    df['next_visit_id'] = df['visit_id'].shift(-1)
    
    return df

def detect_regression_errors(df):
    # Detectar errores de regresión
    for i in range(len(df) - 1):
        if (df.loc[i, 'next_process_step'] < df.loc[i, 'process_step']) and (df.loc[i, 'next_visit_id'] == df.loc[i, 'visit_id']):
            df.loc[i, 'regr_error'] = True
    
    # Calcular la tasa de error general
    error_rate = df['regr_error'].sum() / len(df) * 100
    return error_rate, df

def calculate_error_rate_by_variation(df):
    # Calcular la tasa de error por variación
    error_rate_by_variation = df.groupby('variation').apply(
        lambda x: x['regr_error'].sum() / len(x) * 100
    ).reset_index(name='error_rate')
    
    # Renombrar columnas para mayor claridad
    error_rate_by_variation.columns = ['variation', 'error_rate']
    
    return error_rate_by_variation


def filter_no_errors(df):
    """
    Filtra el DataFrame para incluir solo las filas sin errores de regresión.
    """
    return df[df['regr_error'] == False]

def calculate_average_time_per_variation(df):
    """
    Calcula el tiempo promedio de transiciones válidas por cada variación.
    """
    time_data = []

    # Iterar por cada variación
    for variation, group in df.groupby('variation'):
        # Filtrar los pasos para asegurarse de que solo se consideren los válidos
        filtered_group = group[group['process_step'].isin([0, 1, 2, 3, 4])]

        total_time_var = 0
        valid_transitions = 0

        # Revisar la secuencia de pasos
        for i in range(len(filtered_group) - 1):
            # Verificar si hay una transición válida entre pasos consecutivos
            if (filtered_group.iloc[i]['process_step'] + 1 == filtered_group.iloc[i + 1]['process_step']):
                total_time_var += filtered_group.iloc[i]['time_in_step']
                valid_transitions += 1

        # Solo almacenar si hay transiciones válidas
        if valid_transitions > 0:
            average_time = total_time_var / valid_transitions
            time_data.append({'variation': variation, 'average_time': average_time})

    return time_data

def create_average_time_dataframe(time_data):
    """
    Convierte la lista de tiempo promedio por variación en un DataFrame.
    """
    return pd.DataFrame(time_data)

def filter_no_errors_and_outliers(df):
    """
    Filtra las filas sin errores de regresión y elimina outliers en la columna 'time_in_step' utilizando el método del IQR.
    """
    # Filtrar las filas sin errores
    df_no_errors = df[df['regr_error'] == False]
    
    # Identificar outliers usando el IQR para la columna 'time_in_step'
    Q1 = df_no_errors['time_in_step'].quantile(0.25)
    Q3 = df_no_errors['time_in_step'].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filtrar los outliers
    df_filtered = df_no_errors[(df_no_errors['time_in_step'] >= lower_bound) & (df_no_errors['time_in_step'] <= upper_bound)]
    
    return df_filtered

def calculate_average_time_per_variation(df):
    """
    Calcula el tiempo promedio de transiciones válidas entre pasos consecutivos por variación.
    """
    time_data = []

    # Iterar a través de los grupos por variación
    for variation, group in df.groupby('variation'):
        # Filtrar los pasos de interés (0 a 4)
        filtered_group = group[group['process_step'].isin([0, 1, 2, 3, 4])]

        total_time_step = 0
        valid_transitions = 0

        # Revisar la secuencia de pasos
        for i in range(len(filtered_group) - 1):
            # Verificar si hay una transición válida entre pasos consecutivos
            if filtered_group.iloc[i]['process_step'] + 1 == filtered_group.iloc[i + 1]['process_step']:
                total_time_step += filtered_group.iloc[i]['time_in_step']
                valid_transitions += 1

        # Solo almacenar si hay transiciones válidas
        if valid_transitions > 0:
            average_time = total_time_step / valid_transitions
            time_data.append({'variation': variation, 'average_time': average_time})

    return time_data

def create_average_time_dataframe(time_data):
    """
    Convierte la lista de tiempos promedio por variación en un DataFrame.
    """
    return pd.DataFrame(time_data)





import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm
from sklearn.preprocessing import PowerTransformer, StandardScaler, QuantileTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.decomposition import PCA, FastICA

def binarize_variables(data):
    data['acousticness'] = (data['acousticness'] > 0.05).astype(int)
    data['instrumentalness'] = (data['instrumentalness'] > 0.005).astype(int)
    print("Debug stage 1 :(")
    print(data)
    return data

def drop_variables(data):
    data = data.drop('label', axis = 1, inplace = True)
    data = data.drop(['acousticness', 'instrumentalness', 'key', 'mode', 'time_signature'], axis = 1, inplace = True)
    print("Debug stage 2")
    print(data)
    return data

def concat_variables(transformed_data, original_data, columns_to_keep):
    data = pd.concat([transformed_data.reset_index(drop=True), original_data[columns_to_keep].reset_index(drop=True)], axis=1)
    print("Debug stage 3")
    print(data)
    return data

def compute_missing_data(data, method, testing = False, percentage = 0.15):
    """
    Compute missing data in a DataFrame using specified methods.

    Parameters:
    data (pd.DataFrame): The input DataFrame containing data with potential missing values.
    method (str): The method to compute missing values. Options are 'Mean', 'Median', 'Mode', 'Arbitrary', 'KNN', 'MICE'.
    testing (bool, optional): If True, randomly set a percentage of data to NaN for testing purposes. Default is False.
    percentage (float, optional): The percentage of data to be set to NaN when testing is True. Default is 0.15.

    Returns:
    pd.DataFrame: A DataFrame with missing values computed according to the specified method.
    """
    modified_data = data.copy()
    
    if testing:
        for column in data:
            # Seleccionar aleatoriamente los indices para computar los valores faltantes
            indices_to_null = np.random.choice(data.index, size=int(len(data) * percentage), replace=False)
            modified_data.loc[indices_to_null, column] = np.nan
    
    # Mean, median, mode y arbitrary pueden ser calculados variable por variable
    if method in ['Mean', 'Median', 'Mode', 'Arbitrary']:
        for column in data:
            # Para cada columna (variable), computar los valores faltantes segun el metodo especificado
            for index in modified_data[modified_data[column].isnull()].index:
                if method == 'Mean':
                    value = data[column].mean()
                elif method == 'Median':
                    value = data[column].median()
                elif method == 'Mode':
                    value = data[column].mode().iloc[0]
                elif method == 'Arbitrary':
                    value = np.random.uniform(data[column].min(), data[column].max())
                else:
                    value = np.nan

                if isinstance(value, np.ndarray):
                    value = value[0]
                value = data[column].dtype.type(value)

                modified_data.loc[index, column] = value
    
    # MICE y KNN requieren el dataframe completo para ser calculados
    elif method in ['KNN', 'MICE']:
        if method == 'KNN':
            imputer = KNNImputer(n_neighbors=5)
            imputed = imputer.fit_transform(modified_data)
            modified_data = pd.DataFrame(imputed, columns = modified_data.columns)
        elif method == 'MICE':
            imputer = IterativeImputer()
            imputed = imputer.fit_transform(modified_data)
            modified_data = pd.DataFrame(imputed, columns = modified_data.columns)
    
    print("Debug stage 4")
    print(modified_data)
    return modified_data

def test_normalidad(data, p_thres = 0.05):
    """
    Test for normality of a dataset using the Kolmogorov-Smirnov test.

    Parameters:
    data (np.ndarray): The dataset to be tested for normality.
    p_thres (float, optional): The p-value threshold to determine normality. Default is 0.05.

    Returns:
    tuple: A tuple containing the normality status ('Normal' or 'No normal') and the p-value.
    """
    mean, std = norm.fit(data)

    if std == 0:
        return 'No normal', 1e-8

    normal = norm(loc = mean, scale = std)
    _, p_value = stats.kstest(data, normal.cdf)

    if p_value > p_thres:
        normality = "Normal"
    else:
        normality = "No normal"
    
    return normality, p_value

def handle_outliers(data, method, imputation_method = 'KNN', winsorization_rate = 0.05):
    """
    Handle outliers in a DataFrame using specified methods.

    Parameters:
    data (pd.DataFrame): The input DataFrame containing data with potential outliers.
    method (str): The method to handle outliers. Options are 'Imputacion', 'Trimming', 'Capping', 'Winsorization'.
    imputation_method (str, optional): The imputation method to use if 'Imputacion' is selected. Default is 'KNN'.
    winsorization_rate (float, optional): The rate to use for winsorization. Default is 0.05.

    Returns:
    pd.DataFrame: A DataFrame with outliers handled according to the specified method.
    """
    original_data = data.copy()
    modified_data = data.copy()

    if method == 'Imputacion':
        for column in modified_data.columns:
            q1 = np.percentile(modified_data[column], 25)
            q3 = np.percentile(modified_data[column], 75)
            iqr = q3 - q1
            lower_limit = q1 - 1.5 * iqr
            upper_limit = q3 + 1.5 * iqr

            modified_data[column] = np.where(modified_data[column] < lower_limit, np.nan, np.where(modified_data[column] > upper_limit, np.nan, modified_data[column]))

        # cambia todos los numeros que esten fuera de los limites superior e inferior a np.nan
        outlierless_data = compute_missing_data(modified_data, imputation_method, False, None)
        return outlierless_data
    
    for column in modified_data.columns:
        q1 = np.percentile(original_data[column], 25)
        q3 = np.percentile(original_data[column], 75)
        iqr = q3 - q1
        lower_limit = q1 - 1.5 * iqr
        upper_limit = q3 + 1.5 * iqr

        if method == 'Trimming':
            # Quita todas las filas de data que tengan numeros fuera de los limites superior e inferior
            mask = (modified_data[column] >= lower_limit) & (modified_data[column] <= upper_limit)
            modified_data = modified_data[mask].reset_index(drop=True)
            outlierless_data = modified_data.loc[mask]
        elif method == 'Capping':
            # Cambia los valores de los numeros que estan fuera de los limites superior e inferior a su respectivo limite
            modified_data[column] = np.where(modified_data[column] < lower_limit, lower_limit, np.where(modified_data[column] > upper_limit, upper_limit, modified_data[column]))
            outlierless_data = modified_data
        elif method == 'Winsorization':
            lower_winsor = np.percentile(modified_data[column], 100 * winsorization_rate)
            upper_winsor = np.percentile(modified_data[column], 100 * (1 - winsorization_rate))
            modified_data[column] = np.where(modified_data[column] < lower_winsor, lower_winsor, np.where(modified_data[column] > upper_winsor, upper_winsor, modified_data[column]))
            outlierless_data = modified_data
        else:
            print(f"No hubo match {method}")

    print("Debug stage 5")
    print(outlierless_data)
    return outlierless_data

def transform_data(data, method, p_thres):
    """
    Transform data in a DataFrame using specified methods to achieve normality.

    Parameters:
    data (pd.DataFrame): The input DataFrame containing data to be transformed.
    method (str): The method to use for transformation. Options are 'Auto', 'Exp', 'box-cox', 'yeo-johnson', 'Cuartiles'.
    p_thres (float): The p-value threshold to determine normality.

    Returns:
    tuple: A tuple containing the transformed DataFrame and a list of methods used for each column.
    """
    modified_data = data.copy()
    method_used = []

    if method == 'Auto':
        for column in modified_data.columns:
            methods = ['Exp', 'BoxCox', 'Yeo-Johnson']
            
            # Probamos todas las transformaciones
            # Transformacion exponencial
            exp_data = np.exp(modified_data[column])
            _, p_val_exp = test_normalidad(exp_data, p_thres = p_thres)

            print(exp_data)

            # Transformacion BoxCox
            pt = PowerTransformer(method = 'box-cox')
            if (modified_data[column] <= 0.0).any():
                # BoxCox no se puede hacer con valores negativos asi que le asignamos un valor muy malo para no tenerlo en cuenta
                box_data = modified_data[column]
                p_val_box = 1e-20
            else:
                box_data = pt.fit_transform(modified_data[column].values.reshape(-1, 1))
                _, p_val_box = test_normalidad(box_data, p_thres = p_thres)

            print(box_data)

            # Transformacion Yeo-Johnson
            pt = PowerTransformer(method = 'yeo-johnson')
            yeo_data = pt.fit_transform(modified_data[column].values.reshape(-1, 1))
            _, p_val_yeo = test_normalidad(yeo_data, p_thres = p_thres)

            print(yeo_data)
            
            # Transformacion Cuartiles
            # IMPORTANTE: Ignoramos este metodo porque es medio 'trampa', ya que fuerza todo a una distribucion normal y solo funciona para grandes cantidades de datos
            #qt = QuantileTransformer(n_quantiles = 100, output_distribution='normal')
            #_, p_val_qt = test_normalidad(qt.fit_transform(modified_data[[column]]), p_thres = p_thres)

            data = [exp_data, box_data, yeo_data]

            print(data)

            p_vals = [p_val_exp, p_val_box, p_val_yeo]
            index = np.argmax(p_vals)
            modified_data[column] = data[index]
            method_used.append(methods[index])
        print("Debug stage 6")
        print(modified_data)
        return modified_data
    else:
        for column in modified_data.columns:
            # Si la variable ya es normal, no hace falta transformarla
            if test_normalidad(modified_data[column], p_thres = p_thres) == 'Normal':
                continue
                
            if method == 'Exp':
                if (modified_data[column] > 20).any():
                    print('Los numeros son demasiado grandes para hacer transformacion exponencial')
                    method_used.append('none')
                    continue
                modified_data[column] = np.exp(modified_data[column])
                method_used.append('Exp')
            elif method == 'box-cox':
                if (modified_data[column] <= 0.0).any():
                    print('Hay numeros negativos o cero, no se puede hacer box-cox')
                    method_used.append('none')
                    continue
                pt = PowerTransformer(method = method)
                modified_data[column] = pt.fit_transform(modified_data[column].values.reshape(-1, 1))
                method_used.append('box-cox')
            elif method == 'yeo-johnson':
                pt = PowerTransformer(method = method)
                modified_data[column] = pt.fit_transform(modified_data[column].values.reshape(-1, 1))
                method_used.append('Yeo-Johnson')
            elif method == 'Cuartiles':
                qt = QuantileTransformer(n_quantiles = 100, output_distribution='normal')
                modified_data[column] = qt.fit_transform(modified_data[[column]])
                method_used.append('Cuartiles')
            else:
                print(f"No hubo match {method}")
                
    print("Debug stage 6")
    print(modified_data)
    return modified_data

def reduce_dimensionality(data, method, corr_thres = 0.95, var_thres = 0.01, normality_thres = 0.01, explained_var = 0.99, do_ica = False, filter_non_normal = False):
    """
    Reduces dimensionality of input data using specified method.

    Parameters:
    - data (DataFrame): Input data containing variables to be processed.
    - method (str): Method for dimensionality reduction. Options are 'Filter' for feature filtering based on variance, normality, and correlation thresholds,
                    and 'Projection' for PCA (Principal Component Analysis) followed optionally by ICA (Independent Component Analysis).
    - corr_thres (float, optional): Threshold for removing highly correlated variables (default is 0.95).
    - var_thres (float, optional): Threshold for minimum variance to retain a variable (default is 0.01).
    - normality_thres (float, optional): Threshold for p-value to consider a variable non-normal (default is 0.01).
    - explained_var (float, optional): Minimum variance ratio to be explained by PCA components (default is 0.99).
    - do_ica (bool, optional): Whether to perform Independent Component Analysis (ICA) after PCA (default is False).
    - filter_non_normal (bool, optional): Whether to filter out variables that are non-normal based on normality_thres (default is False).

    Returns:
    - DataFrame: Modified data after dimensionality reduction based on the chosen method.
    """
    #modified_data = data.copy()

    if method == 'Filter':
        # Remover variables que tienen varianza muy pequeña o son muy no normales
        for column in data.columns:
            var = data[column].var()
            
            _, p_val = test_normalidad(data[column], normality_thres)

            if var < var_thres:
                print(f'Quitando {column} por varianza {var}')
                data.drop(column, axis=1, inplace=True)
            elif p_val < normality_thres and filter_non_normal:
                print(f'Quitando {column} por no normal {p_val}')
                data.drop(column, axis=1, inplace=True)

        # Remover variables de alta correlacion
        corr_matrix = np.corrcoef(data, rowvar=False)
        # Llenar la diagonal de la matriz de correlacion de ceros para evitar considerar autocorrelacion
        np.fill_diagonal(corr_matrix, 0)
        corr_mask = np.abs(corr_matrix) < corr_thres
        data = data.loc[:, np.all(corr_mask, axis=0)]

    elif method == 'Projection':
        # Before performing PCA, data should be scaled
        scaler = StandardScaler()
        data = scaler.fit_transform(data)

        # Perform PCA
        pca = PCA(explained_var)
        data = pca.fit_transform(data)
        print(f'Explained variance ratio: {pca.explained_variance_ratio_}')

        # Perform ICA
        if pca.n_components_ > 1 and do_ica:
            ica = FastICA()
            data = ica.fit_transform(data)

        # Transform back into a DataFrame
        data = pd.DataFrame(data, columns=[f"PC{i}" for i in range(1, pca.n_components_ + 1)])
    else:
        print(f"No hubo match {method}")

    print("Debug stage 7")
    print(data)
    return data

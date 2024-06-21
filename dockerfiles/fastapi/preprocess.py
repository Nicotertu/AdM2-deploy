from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
import custom_library as cl
import pandas as pd

def preprocess(original_data):
    modified_data = original_data.copy()
    columns_to_keep = ['acousticness', 'instrumentalness', 'key', 'mode', 'time_signature']

    # Step 1: 
    binarization = FunctionTransformer(lambda data: cl.binarize_variables(data))

    # Step 2: 
    removal = FunctionTransformer(lambda data: data.drop(columns=columns_to_keep))

    # Step 3: Handle outliers and missing data
    handle_outliers = FunctionTransformer(lambda data: cl.handle_outliers(data, method='Imputacion', imputation_method='MICE', winsorization_rate=None))
    imputation = FunctionTransformer(lambda data: cl.compute_missing_data(data, method='KNN', testing=False, percentage=0.05))

    # Step 4: Transformation of variables
    normalization = FunctionTransformer(lambda data: cl.transform_data(data, method='yeo-johnson', p_thres=0.05))

    # Step 5: Dimensionality reduction
    # dimensionality = FunctionTransformer(lambda data: cl.reduce_dimensionality(data, method='Projection', corr_thres=0.8, var_thres=0.01, normality_thres=0.05, explained_var=0.93, do_ica=False))

    # Step 6: 
    def add_columns_back(modified_data, original_data, columns_to_keep):
        modified_data = pd.DataFrame(modified_data, columns=[col for col in original_data.columns if col not in columns_to_keep])
        result = pd.concat([modified_data.reset_index(drop=True), original_data[columns_to_keep].reset_index(drop=True)], axis=1)
        return result
    addition = FunctionTransformer(lambda data: add_columns_back(data, original_data, columns_to_keep), validate=False)

    pipeline = Pipeline(steps=[
        ('binarization', binarization),
        ('removal', removal),
        ('handle_outliers', handle_outliers),
        ('imputation', imputation),
        ('normalization', normalization),
        ('addition', addition)
    ])

    return pipeline.fit_transform(original_data)
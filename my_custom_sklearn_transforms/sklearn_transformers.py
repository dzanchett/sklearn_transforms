from sklearn.base import BaseEstimator, TransformerMixin


# All sklearn Transforms must have the `transform` and `fit` methods
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        import pandas as pd
        import numpy as np
        self.columns = columns

    def fit(self, X, y=None):
        import pandas as pd
        import numpy as np
        d = X.copy()
        self.f = pd.Series([d[column].value_counts().index[0]
            if d[column].dtype == np.dtype('O') else d[column].mean() for column in d],
            index=d.columns)

        return self
    
    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        data = data.drop(labels=self.columns, axis='columns')
        return data.fillna(self.f)

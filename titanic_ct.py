import pandas as pd
import numpy as np

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

class CustomNumImputer(TransformerMixin, BaseEstimator):
    
    def __init__(self):
        self.imputer = None

    def fit(self, X, y=None):
        self.imputer = SimpleImputer(strategy='median')
        self.imputer.fit(X)
        return self

    def transform(self, X):
        X_imputed = pd.DataFrame(self.imputer.transform(X), index=X.index, columns=X.columns)
        return X_imputed


class AddCustomColumns(TransformerMixin, BaseEstimator):
    
    def __init__(self):
        self.features = None
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X['FamSize'] = X['Parch'] + X['SibSp']
        X['IsAlone'] = X['FamSize'].apply(lambda x: 1 if x==0 else 0)
        self.features = X.columns
        return X
    
    def get_feature_names(self):
        return self.features


class CustomScaler(TransformerMixin, BaseEstimator):
    
    def __init__(self):
        self.scaler = None
        self.features = None

    def fit(self, X, y=None):
        self.scaler = MinMaxScaler()
        self.scaler.fit(X)
        self.features = X.columns
        return self

    def transform(self, X):
        X_scaled = pd.DataFrame(self.scaler.transform(X), index=X.index, columns=X.columns)
        return X_scaled
    
    def get_feature_names(self):
        return self.features
    

class CustomCatImputer(TransformerMixin, BaseEstimator):
    
    def __init__(self):
        self.imputer = None

    def fit(self, X, y=None):
        self.imputer = SimpleImputer(strategy='most_frequent')
        self.imputer.fit(X)
        return self

    def transform(self, X): 
        X_imputed = pd.DataFrame(self.imputer.transform(X), index=X.index, columns=X.columns)
        return X_imputed

    
class CustomOneHotEncoder(TransformerMixin, BaseEstimator):
    
    def __init__(self):
        self.features = None
        self.enc = None
    
    def fit(self, X, y=None):
        self.enc = OneHotEncoder(drop="first")
        self.enc.fit(X)
        self.features = self.enc.get_feature_names()
        return self
    
    def transform(self, X):
        X_encoded = pd.DataFrame(self.enc.transform(X).toarray(), columns=self.features)
        return X_encoded
    
    def get_feature_names(self):
        return self.features

    
class CustomTransformerPipeline(Pipeline):
    
    def get_feature_names(self):
        feature_names = []
        for name in self.steps[-1][1].get_feature_names():
            feature_names.append(name.split('__')[-1])
        return feature_names

    
class CustomFeaturePipeline(Pipeline):
    
    def get_feature_names(self):
        feature_names = np.array(self.named_steps['feat_union'].get_feature_names())
        selected_features = feature_names[self.named_steps['feat_selector'].get_support(indices=True)]
        selected_features_pretty = []
        for feature in selected_features:
            selected_features_pretty.append(feature.split('__')[-1])
        return selected_features_pretty


if __name__ == '__main__':
	print('mymodule is being run directly!')
else:
	print('mymodule is being imported')
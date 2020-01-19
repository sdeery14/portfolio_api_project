import pandas as pd
import numpy as np
from joblib import dump

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import RFE

from titanic_ct import *

if __name__ == '__main__':

	train = pd.read_csv('train.csv')

	X_train = train[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex', 'Embarked']].copy()
	y_train = train['Survived']
	num_feats = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
	cat_feats = ['Sex', 'Embarked']

	feat_pipe = CustomFeaturePipeline([('feat_union', FeatureUnion([
	    ('num_trans', ColumnTransformer(transformers=[
	        ('num_pipe', CustomTransformerPipeline([
	            ('imputer', CustomNumImputer()), 
	            ('add_cols', AddCustomColumns()),
	            ('scaler', CustomScaler())]), 
	        num_feats)])), 
	    ('cat_trans', ColumnTransformer(transformers=[
	        ('cat_pipe', CustomTransformerPipeline([
	            ('imputer', CustomCatImputer()),
	            ('one_hot', CustomOneHotEncoder())]), 
	        cat_feats)]))])),
                                  ('feat_selector', RFE(RandomForestClassifier(n_estimators=100)))])

	predictor = Pipeline([('feat_pipe', feat_pipe), ('clf', GradientBoostingClassifier(n_estimators=200))]).fit(X_train, y_train)

	dump(predictor, 'titanic_predictor.joblib')

else:
	print('myprogram is being imported')

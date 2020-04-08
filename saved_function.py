import os
from collections import defaultdict
import inspect
import pandas as pd
import numpy as np
from scipy import stats
import re
from graphviz import Digraph
import plotly
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import json
from functools import wraps

from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize, KBinsDiscretizer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from utils import *
from fairness_instru import *
@tracer(cat_col = ['sex', 'race'], numerical_col = ['age', 'hours-per-week'], sensi_atts=['sex', 'race'], target_name = "income-per-year", training=True, save_path="case_outputs/webUI", dag_save="svg")
def adult_pipeline_normal(f_path = 'data/adult_train.csv'):
    data = pd.read_csv(f_path, na_values='?', index_col=0)
#     data = raw_data.dropna()

    labels = label_binarize(data['income-per-year'], ['>50K', '<=50K'])

    nested_categorical_feature_transformation = Pipeline(steps=[
        ('impute', SimpleImputer(missing_values=np.nan, strategy='most_frequent')),
        ('encode', OneHotEncoder(handle_unknown='ignore'))
    ])

    nested_feature_transformation = ColumnTransformer(transformers=[
        ('categorical', nested_categorical_feature_transformation, ['education', 'workclass']),
        ('numeric', StandardScaler(), ['age', 'hours-per-week'])
    ])

    nested_pipeline = Pipeline([
      ('features', nested_feature_transformation),
      ('classifier', DecisionTreeClassifier())])

    return nested_pipeline
pipeline = adult_pipeline_normal()
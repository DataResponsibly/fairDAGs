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
@tracer(cat_col = ['sex', 'race'], numerical_col = ['age', 'hours-per-week'], sensi_atts=['sex', 'race'], target_name = "income-per-year", training=True, save_path="experiments/510e946c-83f8-11ea-84e2-acde48001122", dag_save="svg")
def adult_pipeline_easy(f_path = "./media/510e946c-83f8-11ea-84e2-acde48001122.csv"):

    raw_data = pd.read_csv(f_path, na_values='?', index_col=0)

    data = raw_data.dropna()

    labels = label_binarize(data['income-per-year'], ['>50K', '<=50K'])

    feature_transformation = ColumnTransformer(transformers=[
        ('categorical', OneHotEncoder(handle_unknown='ignore'), ['education', 'workclass']),
        ('numeric', StandardScaler(), ['age', 'hours-per-week'])
    ])


    income_pipeline = Pipeline([
        ('features', feature_transformation),
        ('classifier', DecisionTreeClassifier())])

    return income_pipeline
pipeline = adult_pipeline_easy()
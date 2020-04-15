from flask import Flask, render_template, redirect, url_for, request, session, flash, Markup
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
import re
import uuid
from functools import wraps
from importlib import reload

from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize, KBinsDiscretizer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from utils import *
from fairness_instru import *

import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_colwidth',1000)
np.set_printoptions(precision = 4)
pd.set_option("display.precision", 4)
pd.set_option('expand_frame_repr', True)

app = Flask(__name__)

cache = []

user_id = uuid.uuid1()

essentials = """import os
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
"""

# variable initilization
target_name, pos_group, code, name, organization = "", "", "", "", ""
num_target, cat_target = [], []

cache = []
log_dict = {}
plot_dict = {}
rand_rgb = {}

# flask secret_key
app.secret_key = "shdgfashfasdsfsdf"

def login_required(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if 'logged_in' in session:
            return f(*args, **kwargs)
        else:
            flash('You need to login first')
            return redirect(url_for('login'))
    return wrap

@app.route('/login', methods = ['GET', 'POST'])
def login():
    error = None

    # set default_value here
    if request.method == 'POST':
        global name
        name = request.form['name'] if request.form['name'] else 'Guest'
        global organization
        organization = request.form['organization'] if request.form['organization'] else 'Unknown'\
        global demo
        demo = request.form['demo']
        global code
        code = """def adult_pipeline_easy(f_path = 'playdata/AD_train.csv'):

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

    return income_pipeline""" if not request.form['code'] else request.form['code']
        global target_name, pos_group
        target_name, pos_group = list(map(lambda x: x.strip(), request.form['target_name'].split(','))) if request.form['target_name'] else ("income-per-year", ">50K")
        global cat_target
        cat_target = list(map(lambda x: x.strip(), request.form['cat_target'].split(','))) if request.form['cat_target'] else ['sex', 'race']
        global num_target
        num_target = list(map(lambda x: x.strip(), request.form['num_target'].split(','))) if request.form['num_target'] else ['age', 'hours-per-week']
        global save_path
        save_path = f'experiments/{user_id}'
        global perform_target
        perform_target = request.form['perform_target'] if request.form['perform_target'] else 'PR'

        session['logged_in'] = True

        global cache
        cache = []
        global log_dict
        global rand_rgb
        global plot_dict
        global target_df


        flash('You were just logged in')
        if not demo == 'USER':
            log_dict = pickle.load(open(f"playdata/{demo}/checkpoints/log_dict_train.p", 'rb'))
            rand_rgb = pickle.load(open(f"playdata/{demo}/checkpoints/rand_color_train.p", 'rb'))
            rand_rgb = pickle.load(open(f"playdata/{demo}/checkpoints/rand_color_train.p", 'rb'))
            plot_dict = pickle.load(open(f"playdata/{demo}/checkpoints/plot_dict_train.p", 'rb'))
            target_df = pickle.load(open(f"playdata/{demo}/checkpoints/target_df_train.p", 'rb'))

            with open(f"playdata/{demo}/DAG/pipeline.svg", 'r') as content:
                svg = content.read()
            with open(f'templates/{demo}.html', 'w+') as f:
                f.write("{% extends 'index1.html' %}\n")
                f.write("{% block content %}\n")
                f.write(svg)
                f.write('\n')
                f.write("{% endblock %}\n")
            return redirect(url_for('home'))


        function_title = code.split('(')[0].replace('def ','')+"()"
        with open(f'{user_id}.py', 'w+') as f:
            f.write(essentials)
            f.write(f"""@tracer(cat_col = {cat_target}, numerical_col = {num_target}, sensi_atts={cat_target}, target_name = \"{target_name}\", training=True, save_path=\"{save_path}\", dag_save=\"svg\")\n{code}\n""")
            f.write(f"pipeline = {function_title}")
        os.system(f"python {user_id}.py")

        img = save_path + "/DAG/pipeline.svg"
        with open(img, 'r') as content:
            svg = content.read()
        with open(f'templates/{user_id}.html', 'w+') as f:
            f.write("{% extends 'index1.html' %}\n")
            f.write("{% block content %}\n")
            f.write(svg)
            f.write('\n')
            f.write("{% endblock %}\n")

        log_dict = pickle.load(open(save_path+"/checkpoints/log_dict_train.p", 'rb'))
        rand_rgb = pickle.load(open(save_path+"/checkpoints/rand_color_train.p", 'rb'))
        rand_rgb = pickle.load(open(save_path+"/checkpoints/rand_color_train.p", 'rb'))
        plot_dict = pickle.load(open(save_path+"/checkpoints/plot_dict_train.p", 'rb'))
        target_df = pickle.load(open(save_path+"/checkpoints/target_df_train.p", 'rb'))

        return redirect(url_for('home'))
    return render_template("login_2.html", error = error)

@app.route('/', methods=['GET'])
@login_required
def home():
    # print("code"+code)

    selected_status = request.args.get('type')
    if selected_status is not None:
        cache.append(selected_status)

    corr_color = [rand_rgb[int(step)] for step in cache]
    plots = {}
    to_plot = []
    code_with_color = ""
    tables_to_display, titles, labels, code_titles, plt_xs, plt_ys, plt_titles, plt_xas, plt_yas, plot_log_changes = [], [], [], [], [], [], [], [], [], []
    for status in cache:
        if 'Classifier' in int_to_string(int(status)):
            label_inverse = {1: '<=50K', 0:'>50K'}
            target_df[target_name].replace(label_inverse, inplace = True)
            target_df['pred_'+target_name].replace(label_inverse, inplace = True)
            plt_titles.insert(0, 'Performance Label')
            to_plot.insert(0, (get_performance_label(target_df, cat_target, target_name, pos_group), perform_target))
            to_plot.append(static_label(target_df, cat_target, target_name))
            plot_log_changes.append(pd.DataFrame(static_label(target_df, cat_target, target_name)))
        else:
            to_plot.append(sort_dict_key(plot_dict[int(status)]))
            plot_log_changes.append(pd.DataFrame(sort_dict_key(plot_dict[int(status)])))
        if int(status) in log_dict.keys():
            temp_table = log_dict[int(status)]

            if len(plot_log_changes) == 1:
                tables_to_display.append('No changes')
            else:
                if plot_log_changes[-1].equals(plot_log_changes[-2]):
                    tables_to_display.append('No changes')
                else:
                    tables_to_display.append((plot_log_changes[-1] - plot_log_changes[-2]).to_html(classes = 'table table-striped'))

            for key, dataframe in temp_table.items():

                tables_to_display.append(dataframe.to_html(classes = 'table table-striped'))
                num_cat = "NUMERICAL features" if key == 'num' else "CATEGORICAL features"
                titles.append(int_to_string(int(status)))
                labels.append(' -- Static Label, show changes in percentage')
                titles.append(int_to_string(int(status)))
                labels.append(" -- TARGET changed in "+num_cat)
                code_titles.append(int_to_string(int(status)))

                # start_plotly
                if key == 'cat':
                    plt_titles.append('INSPECTING ' + int_to_string(int(status)))
                else:
                    plt_titles.append('INSPECTING ' + int_to_string(int(status)))
        else:

            if len(plot_log_changes) == 1:
                tables_to_display.append('No changes')
            else:
                if plot_log_changes[-1].equals(plot_log_changes[-2]):
                    tables_to_display.append('No changes')
                else:
                    tables_to_display.append((plot_log_changes[-1] - plot_log_changes[-2]).to_html(classes = 'table table-striped'))
            tables_to_display.append('No changes')
            titles.append(int_to_string(int(status)))
            labels.append(' -- Static Label, show changes in percentage')
            titles.append(int_to_string(int(status)))
            labels.append('')
            code_titles.append(int_to_string(int(status)))

            plt_titles.append('INSPECTING ' + int_to_string(int(status)))

        plots = create_hist_sub_plot(to_plot[::-1], plt_titles[::-1], pos_group)

    code_with_color = change_code_color(corr_color, code_titles, code)
    template_to_render = user_id if demo=="USER" else demo
    return render_template(f'{template_to_render}.html', svg = svg, plots = plots, tables = tables_to_display[::-1], titles = titles[::-1], labels = labels[::-1], colors = np.array(corr_color[::-1]).repeat(2).tolist(), code = code_with_color, name = name, org = organization)

@app.route('/logout')
@login_required
def logout():
    session.pop('logged_in', None)
    flash('You were just logged out')
    return redirect(url_for('login'))

if __name__ == '__main__':
   app.run(debug=True)

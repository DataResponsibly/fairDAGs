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

def int_to_string(n):
    return n.to_bytes(math.ceil(n.bit_length() / 8), 'little').decode()

# def create_sub_plot(plt_xs, plt_ys, plt_xas, plt_yas, plt_titles, colors):
#     no_rows = len(plt_xs)
#     no_rows_plot = no_rows//2 if no_rows%2==0 else no_rows//2+1
#     fig = make_subplots(
#         rows=no_rows_plot, cols=2, subplot_titles=plt_titles
#     )
#
#     for i, point in enumerate(zip(plt_xs, plt_ys)):
#         fig.add_trace(go.Bar(x=point[0], y=point[1], text=point[1], textposition='auto', marker_color = colors[i]), row=i//2+1, col=i%2+1)
#
#     for i, x_axis in enumerate(plt_xas):
#         fig.update_xaxes(title_text=x_axis, row=i//2+1, col=i%2+1)
#
#     for i, y_axis in enumerate(plt_yas):
#         fig.update_yaxes(title_text=y_axis, row=i//2+1, col=i%2+1)
#
#     fig.update_layout(
#         height = 450*no_rows_plot,
#         font=dict(
#             family="Courier New, monospace",
#             size=14,
#             color="#7f7f7f"
#             )
#         )
#
#     graphJSON = json.dumps(fig, cls = plotly.utils.PlotlyJSONEncoder)
#     return graphJSON

def create_hist_sub_plot(to_plot, plt_titles, pos_group):
    no_rows = len(to_plot)
    # no_rows_plot = no_rows//2 if no_rows%2==0 else no_rows//2+1

    fig = make_subplots(
        rows=no_rows, cols=2, subplot_titles=np.array(plt_titles[:-1]).repeat(2).tolist()+[plt_titles[-1]]
    )

    for i, res in enumerate(to_plot):
        if type(res) == tuple:
            bar_list = plotly_histogram_perform(data = res[0], target = res[1], title = plt_titles[i])
            for bar_trace in bar_list:
                fig.add_trace(bar_trace, row=i+1, col = 1)
        else:
            bar_list, No_col = plotly_histogram(res, plt_titles[i], i==0, pos_group)
            for j, bar_trace in enumerate(bar_list):
                if j < No_col:
                    fig.add_trace(bar_trace, row=i+1, col=1)
                else:
                    fig.add_trace(bar_trace, row=i+1, col=2)

    for i in range(2 * no_rows):
        fig.update_yaxes(title_text='Positive' if i%2==0 else 'Negative', row=i//2+1, col=i%2+1)

    fig.update_layout(
        legend = dict(
            orientation = 'h',
        ),
        height = 450*no_rows,
        font=dict(
            family="Courier New, monospace",
            size=24,
            color="#7f7f7f"
            )
    )

    graphJSON = json.dumps(fig, cls = plotly.utils.PlotlyJSONEncoder)
    return graphJSON

def change_code_color(colors, titles, code):
    code_list = code.split("\n")

    for idx, line in enumerate(code_list):
        if line.startswith("            "):
            code_list[idx] = f"<p style=\"margin-left: 120px\">{line}</p>"
        elif line.startswith("        "):
            code_list[idx] = f"<p style=\"margin-left: 80px\">{line}</p>"
        elif line.startswith("    "):
            code_list[idx] = f"<p style=\"margin-left: 40px\">{line}</p>"
        elif line=='\r':
            code_list[idx] = f"<p></p>"
        for i, title in enumerate(titles):
            if '__' not in title or title.startswith('__'):
                if title.strip("__") in line:
                    code_list[idx] = f"<font color=\"{colors[i]}\">{code_list[idx]}</font>"
            else:
                # for sep in title.split("__"):
                    # if sep in line and sep:
                if title.split('__')[-2] in line:
                    code_list[idx] = f"{code_list[idx].split(title.split('__')[-2])[0]}<font color=\"{colors[i]}\">{title.split('__')[-2]}</font>{code_list[idx].split(title.split('__')[-2])[-1]}"

        code = "".join(code_list)

    return code

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

target_name, pos_group, code, name, organization = "", "", "", "", ""
num_target, cat_target = [], []

cache = []
log_dict = {}
plot_dict = {}
rand_rgb = {}

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
        organization = request.form['organization'] if request.form['organization'] else 'Unknown'
        global code
        code = """def adult_pipeline_easy(f_path = 'data/adult_train.csv'):

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
        save_path = 'case_outputs/'+request.form['path'] if request.form['path'] else 'case_outputs/adult_easy'
        global perform_target
        perform_target = request.form['perform_target'] if request.form['perform_target'] else 'PR'

        session['logged_in'] = True
        function_title = code.split('(')[0].replace('def ','')+"()"
        with open('saved_function.py', 'w+') as f:
            f.write(essentials)
            f.write(f"""@tracer(cat_col = {cat_target}, numerical_col = {num_target}, sensi_atts={cat_target}, target_name = \"{target_name}\", training=True, save_path=\"{save_path}\", dag_save=\"svg\")\n{code}\n""")
            f.write(f"pipeline = {function_title}")
        os.system("python saved_function.py")
        global cache
        cache = []
        global log_dict
        log_dict = pickle.load(open(save_path+"/checkpoints/log_dict_train.p", 'rb'))
        global rand_rgb
        rand_rgb = pickle.load(open(save_path+"/checkpoints/rand_color_train.p", 'rb'))
        global plot_dict
        plot_dict = pickle.load(open(save_path+"/checkpoints/plot_dict_train.p", 'rb'))
        global target_df
        target_df = pickle.load(open(save_path+"/checkpoints/target_df_train.p", 'rb'))

        flash('You were just logged in')
        return redirect(url_for('home'))
    return render_template("login 2.html", error = error)

@app.route('/', methods=['GET'])
@login_required
def home():
    # print("code"+code)
    img = save_path + "/DAG/pipeline.svg"
    with open(img, 'r') as content:
        svg = content.read()
    with open('templates/index.html', 'w+') as f:
        f.write("{% extends 'index1.html' %}\n")
        f.write("{% block content %}\n")
        f.write(svg)
        f.write('\n')
        f.write("{% endblock %}\n")

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
                        # print(plot_log_changes[-1])
                        # print()
                        # print(plot_log_changes[-2])
                        # print()

                # tables_to_display.append(pd.DataFrame(plot_dict[int(status)]).to_html(classes = 'table table-striped'))
                tables_to_display.append(dataframe.to_html(classes = 'table table-striped'))
                num_cat = "NUMERICAL features" if key == 'num' else "CATEGORICAL features"
                titles.append(int_to_string(int(status)))
                labels.append(' -- Static Label, show changes in percentage')
                titles.append(int_to_string(int(status)))
                labels.append(" -- TARGET changed in "+num_cat)
                code_titles.append(int_to_string(int(status)))

                # start_plotly
                if key == 'cat':
                    # initial plots - population
                    # df_to_plot = pd.DataFrame.from_dict(temp_table[key]['class_percent'][0], orient='index', columns=['percentage'])
                    # plt_xs.append(df_to_plot.index)
                    # plt_ys.append(df_to_plot['percentage'].values)
                    # plt_xas.append('categories')
                    # plt_yas.append('change in percentage')
                    # plt_titles.append('INSPECTING' + int_to_string(int(status))+' Change on '+str(temp_table[key].index[0]))
                    plt_titles.append('INSPECTING ' + int_to_string(int(status)))
                else:
                    # initial plots - population
                    # df_to_plot = temp_table[key]
                    # plt_xs.append(df_to_plot.columns)
                    # plt_ys.append(df_to_plot.iloc[0,:].values)
                    # plt_xas.append('stats')
                    # plt_yas.append('change in counts')
                    # plt_titles.append(int_to_string(int(status))+' Change on '+str(temp_table[key].index[0]))
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

            # initial plots - population
            # df_to_plot = pd.DataFrame({'blank_output_no_changes':0.0}, index=[0])
            #
            # plt_xs.append(df_to_plot.columns)
            # plt_ys.append([0.0])
            # plt_xas.append('')
            # plt_yas.append('')
            # plt_titles.append(int_to_string(int(status))+' No Changes Found')
            plt_titles.append('INSPECTING ' + int_to_string(int(status)))

        # plots = create_sub_plot(plt_xs[::-1], plt_ys[::-1], plt_xas[::-1], plt_yas[::-1], plt_titles[::-1], colors = corr_color[::-1])

        plots = create_hist_sub_plot(to_plot[::-1], plt_titles[::-1], pos_group)

    code_with_color = change_code_color(corr_color, code_titles, code)
    return render_template('index.html', svg = svg, plots = plots, tables = tables_to_display[::-1], titles = titles[::-1], labels = labels[::-1], colors = np.array(corr_color[::-1]).repeat(2).tolist(), code = code_with_color, name = name, org = organization)

@app.route('/logout')
@login_required
def logout():
    session.pop('logged_in', None)
    flash('You were just logged out')
    return redirect(url_for('login'))

if __name__ == '__main__':
   app.run(debug=True)

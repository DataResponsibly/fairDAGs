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
import math
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from functools import wraps
from importlib import reload

from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize, KBinsDiscretizer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

np.set_printoptions(precision = 4)
pd.set_option("display.precision", 4)
class DataFlowVertex:
    def __init__(self, parent_vertices, name, operation, params):
        self.parent_vertices = parent_vertices
        self.name = name
        self.operation = operation
        self.params = params

    def __repr__(self):
        return "(vertex_name={}: parent={}, op={}, params={})".format(self.name, self.parent_vertices, self.operation, self.params)

    def display(self):
        print("(vertex_name={}: parent={}, op={}, params={})".format(self.name, self.parent_vertices, self.operation, self.params))


def line_cleansing(line):
    return int.from_bytes(line.encode(), 'little')

def sort_dict_key(dict):
    new_dict = {}
    for key in sorted(dict.keys()):
        new_dict[key] = dict[key]
    return new_dict

def remove_list_dup(sample_list):
    res = []
    for ele in sample_list:
        if ele not in res:
            res.append(ele)
    return res

def plotly_histogram(res, title, ind, pos_group):
    colors = px.colors.qualitative.Plotly
    primary_key = remove_list_dup([item[0] for item in res.keys()])
    names = remove_list_dup([item[1] for item in res.keys()])
    ys_pos, ys_neg = [], []
    for value in res.values():
        neg_group = list(value.keys())[1-list(value.keys()).index(pos_group)]
        ys_pos.append(value[pos_group])
        ys_neg.append(value[neg_group])
    # ys_pos = [ys_pos[i*len(primary_key):(i + 1) * len(primary_key)] for i in range((len(ys_pos) + len(primary_key) - 1) // len(primary_key))]
    ys_pos = np.array(ys_pos).reshape(2, int(len(ys_pos)/2)).T
    # ys_neg = [ys_neg[i*len(primary_key):(i + 1) * len(primary_key)] for i in range((len(ys_neg) + len(primary_key) - 1) // len(primary_key))]
    ys_neg = np.array(ys_neg).reshape(2, int(len(ys_neg)/2)).T
    if ind:
        bars = [go.Bar(name=name, x=primary_key, y=y, legendgroup=str(i), marker_color = colors[i]) for i, (name,y) in enumerate(zip(names, ys_pos))]
        for i, (name,y) in enumerate(zip(names, ys_neg)):
            bars.append(go.Bar(name = name, x=primary_key, y=y, legendgroup=str(i), showlegend=False, marker_color = colors[i]))
    else:
        bars = [go.Bar(name=name, x=primary_key, y=y, legendgroup=str(i), showlegend=False, marker_color = colors[i]) for i, (name,y) in enumerate(zip(names, ys_pos))]
        for i, (name,y) in enumerate(zip(names, ys_neg)):
            bars.append(go.Bar(name = name, x=primary_key, y=y, legendgroup=str(i), showlegend=False, marker_color = colors[i]))

    return bars, len(ys_pos)

def plotly_histogram_perform(data, target, title):
    colors = px.colors.qualitative.Plotly
    primary_key = remove_list_dup([item[0] for item in data.keys()])
    names = remove_list_dup([item[1] for item in data.keys()])
    target_data = []
    for value in data.values():
        target_data.append(value[target])
    target_data = np.array(target_data).reshape(2, int(len(target_data)/2)).T

    bars = [go.Bar(name=name, x=primary_key, y=y, legendgroup=str(i), showlegend=False, marker_color = colors[i]) for i, (name,y) in enumerate(zip(names, target_data))]

    return bars

def pipeline_to_dataflow_graph(pipeline):
    '''
    Support function used in Logs. Parent_vertices are all set to None here.
    name field is set to be column name in ColumnTransfomer and operation field contains not only name but also input args.
    '''
    graph = []
    layer_graph = []
    def helper(pipeline, name_prefix=[], parent_vertices=[]):
        if 'ColumnTransformer' in str(type(pipeline)):
            for step in pipeline.transformers:
                for column_name in step[2]:
                    # helper(step[1], name_prefix+['ColTrans__'+column_name], parent_vertices)
                    helper(step[1], name_prefix+[column_name], parent_vertices)
        elif 'Pipeline' in str(type(pipeline)):
            for i, key in enumerate(pipeline.named_steps.keys()):
                helper(pipeline.named_steps[key], name_prefix, parent_vertices)

        else :
            graph.append(DataFlowVertex(parent_vertices, ''.join(name_prefix), pipeline, None))

    helper(pipeline)
    return graph

def topo_sort(graph):
    adjacency_list = {vertex.name: [] for vertex in graph}
    visited = {vertex.name: False for vertex in graph}

    for vertex in graph:
        for parent_vertex in vertex.parent_vertices:
            adjacency_list[parent_vertex.name].append(vertex.name)

    output = []

    def toposort(vertex_name, adjacency_list, visited, output):
        visited[vertex_name] = True
        for child_name in adjacency_list[vertex_name]:
            if not visited[child_name]:
                toposort(child_name, adjacency_list, visited, output)
        output.append(vertex_name)

    for vertex_name in adjacency_list.keys():
        if not visited[vertex_name]:
            toposort(vertex_name, adjacency_list, visited, output)

    output.reverse()

    vertices_by_name = {vertex.name: vertex for vertex in graph}

    sorted_graph = []
    for vertex_name in output:
        sorted_graph.append(vertices_by_name[vertex_name])
    return sorted_graph


def find_sink(graph):
    sorted_graph = topo_sort(graph)
    return sorted_graph[-1]

def search_vertex_by_names(names, vertices_list):
    result = []
    names_to_drop = set()
    for vertex in set(vertices_list):
        if vertex.name.split('_')[0] in names:
            if '_' in vertex.name:
                names_to_drop.add(vertex.name.split('_')[0])
            result.append(vertex)
    for item in result:
        if item.name in names_to_drop:
            result.remove(item)
    return result

def static_label(data, sensi_atts, target_name):
    groupby_cols = sensi_atts+[target_name]
    placeholder_att = list(set(data.columns).difference(groupby_cols))[0]
    pivot_data = data.pivot_table(index=sensi_atts,
                   columns=target_name,
                   values=placeholder_att,
                   fill_value=0,
                   aggfunc='count').unstack(fill_value=0).stack()

    # pivot_data.loc['Female'].apply(lambda x: )
    first_index = set([item[0] for item in list(pivot_data.index)])
    res = {}
    for idx in first_index:
        sum_to_norm = pivot_data.loc[idx].sum().get_values()
        res.update({(idx, key): value for key, value in round(pivot_data.loc[idx]/sum_to_norm, 3).T.to_dict().items()})
    return res

def func_aggregation(func_str):
    '''
    This function is used for line execution with exec()

    args:
        function strings after inspect
    returns:
        list of functionable strings for exec()
    '''

    res = [] # executables for return
    stack_for_parent = [] # stack storing brackets for line integration
    logs_of_parent = [] # logs of lines for concat
#     convert function codes to list of strings
    func_list = [item.strip() for item in func_str.split('\n')]
    i = 0
    while True:
        if func_list[i].startswith('def'):
            func_list = func_list[i:]
            break
        i = i + 1
#     function args
    func_args = [item.strip() for item in func_list[0].split('(')[1].rstrip('):').split(',')]
    for item in func_list[1:]:
        if not item:
            continue
        logs_of_parent.append(item)
        for char in item:
            if char == '(':
                stack_for_parent.append('(')
            if char == '[':
                stack_for_parent.append('[')
            if char == ')' and stack_for_parent[-1] == '(':
                stack_for_parent.pop(-1)
            if char == ']' and stack_for_parent[-1] == '[':
                stack_for_parent.pop(-1)
        if not stack_for_parent:
            res.append(''.join(logs_of_parent))
            logs_of_parent.clear()
    return func_args, res[:-1], [item.strip() for item in res[-1].replace('return ', '').split(',')]

def handle_dict(dict_1, dict_2):
    '''
    Calculate differences between two dictionaries
    eg: input: d1 = {'a': 1, 'b': 2, 'c': 3, 'e': 2, 'f':4}
               d2 = {'a': 10, 'b': 9, 'c': 8, 'd': 7, 'e': 3}
        output: z = {'a': -9, 'b': -7, 'c': -5, 'e': -1, 'f': 4, 'd': -7}
    '''

    dict_1_re = {key: round(dict_1.get(key,0) - dict_2.get(key, 0), 4) for key in dict_1.keys()}
    dict_2_re = {key: round(dict_1.get(key,0) - dict_2.get(key, 0), 4) for key in dict_1.keys()}
    return {**dict_1_re, **dict_2_re}

def get_categorical_dif(cat_df, cat_metric_list, prev):
    '''
    Calculate differences for categorical dataframe comparison
    Need special handling for 'class_count' and 'class_percent'
        since they are stored as dict in the dataframe
    '''
    cat_dif = pd.DataFrame()
    for i in cat_metric_list:
        if i != 'class_count' and i != 'class_percent':
            # if the metric is not defined as a dictionary
            dif = cat_df[i] - prev[i]
            cat_dif[i] = dif
        else:
            for idx, col in enumerate(cat_df.index):
                dif = handle_dict(cat_df[i][idx], prev[i][idx])
                cat_dif.loc[col, i] = [dif]
    return cat_dif

def cal_numerical(target_df_1, numeric_feature, numerical_df):
    '''
    Calculate metrices for numerical features
        including counts, missing values, Median and MAD, range/scaling
    '''

    # get counts of non NA values
    count_log = target_df_1[numeric_feature].count()
    numerical_df.loc[numeric_feature, 'count'] = count_log

    # get missing value counts
    missing_count_log = target_df_1[numeric_feature].isna().sum()
    numerical_df.loc[numeric_feature, 'missing_count'] = missing_count_log

    # distribution
    # Median and MAD
    median_log = target_df_1[numeric_feature].median()
    numerical_df.loc[numeric_feature, 'median'] = median_log
    if missing_count_log == 0:
        mad_log = stats.median_absolute_deviation(target_df_1[numeric_feature])
        numerical_df.loc[numeric_feature, 'mad'] = mad_log
    else:
        numerical_df.loc[numeric_feature, 'mad'] = 0

    # range/ scaling
    range_log = target_df_1[numeric_feature].max() - target_df_1[numeric_feature].min()
    numerical_df.loc[numeric_feature, 'range'] = range_log

    return numerical_df

def cal_categorical(target_df_1, cat_feature, cat_df):
    '''
    Calculate metrices for categorical features
        including missing values, number of classes, counts for each group, percentage for each group
    '''

    # get missing value counts
    missing_count_log = target_df_1[cat_feature].isna().sum()
    cat_df.loc[cat_feature, 'missing_count'] = missing_count_log

    # get number of classes
    num_class_log = len(target_df_1[cat_feature].value_counts().keys())
    cat_df.loc[cat_feature, 'num_class'] = num_class_log

    # get counts for each group
    class_count_log = target_df_1[cat_feature].value_counts().to_dict()
    cat_df.loc[cat_feature, 'class_count'] = [class_count_log]

    # get percentage each group covers
    class_percent_log = (target_df_1[cat_feature].value_counts() / \
    target_df_1[cat_feature].value_counts().sum()).to_dict()
    cat_df.loc[cat_feature, 'class_percent'] = [class_percent_log]

    return cat_df

def pipeline_to_dataflow_graph_full(pipeline, name_prefix='', parent_vertices=[]):
    graph = []
    parent_vertices_for_current_step = parent_vertices
    parent_vertices_for_next_step = []

    for step_name, component in pipeline.steps:
        component_class_name = component.__class__.__name__

        if component_class_name == 'ColumnTransformer':
            for transformer_prefix, transformer_component, columns in component.transformers:
                for column in columns:
                    # name = 'ColTrans__'+column
                    name = column
                    transformer_component_class_name = transformer_component.__class__.__name__

                    if transformer_component_class_name == 'Pipeline':

                        vertices_to_add = pipeline_to_dataflow_graph_full(transformer_component,
                                                                     name + "__",
                                                                     parent_vertices_for_current_step)

                        for vertex in vertices_to_add:
                            graph.append(vertex)

                        parent_vertices_for_next_step.append(find_sink(vertices_to_add))

                    else:
                        vertex = DataFlowVertex(parent_vertices_for_current_step,
                                                name_prefix + name,
                                                transformer_component_class_name,
                                               '')
                        graph.append(vertex)
                        parent_vertices_for_next_step.append(vertex)

        else:
            vertex = DataFlowVertex(parent_vertices_for_current_step,
                                    name_prefix + step_name,
                                    component_class_name,
                                   '')
            graph.append(vertex)
            parent_vertices_for_next_step.append(vertex)

        parent_vertices_for_current_step = parent_vertices_for_next_step.copy()
        parent_vertices_for_next_step = []

    return graph


def check_cat_dif(cat_dif_):

    for i in cat_dif_.columns:
        if i != 'class_count' and i != 'class_percent':
            if cat_dif_.loc[:,i].any() !=0:
                return True  # dif exists
        else:
            for sub_dict in cat_dif_.loc[:,i].values:
                if sum(sub_dict.values()) != 0:
                    return True # dif exists
    return False # cat_dif is all zero

def compute_evaluation_metric_binary(true_y, pred_y, label_order):
    TN, FP, FN, TP = confusion_matrix(true_y, list(pred_y), labels=label_order).ravel()
    P = TP + FN
    N = TN + FP
    ACC = (TP+TN) / (P+N) if (P+N) > 0.0 else np.float64(0.0)
    return dict(
                PR = P/ (P+N), P = TP + FN, N = TN + FP,
                TPR=TP / P, TNR=TN / N, FPR=FP / N, FNR=FN / P,
                PPV=TP / (TP+FP) if (TP+FP) > 0.0 else np.float64(0.0),
                NPV=TN / (TN+FN) if (TN+FN) > 0.0 else np.float64(0.0),
                FDR=FP / (FP+TP) if (FP+TP) > 0.0 else np.float64(0.0),
                FOR=FN / (FN+TN) if (FN+TN) > 0.0 else np.float64(0.0),
                ACC=ACC,
                ERR=1-ACC,
                F1=2*TP / (2*TP+FP+FN) if (2*TP+FP+FN) > 0.0 else np.float64(0.0)
            )
def get_performance_label(df, sensi_atts, target_name, posi_target, output_metrics=["TPR", "FPR", "TNR", "FNR", "PR"], round_digit=3):

    groupby_cols = sensi_atts+[target_name]
    placeholder_att = list(set(df.columns).difference(groupby_cols))[0]

    count_all = df[groupby_cols+[placeholder_att]].groupby(groupby_cols).count()
    index_all = list(count_all.index)

    res_dict = {}
    target_label_order = [posi_target, set(df[target_name]).difference([posi_target]).pop()]

    for tuple_i in index_all:
        if len(tuple_i[:-1]) == 1:
            key_tuple = (tuple_i[0])
        else:
            key_tuple = tuple_i[:-1]
        cur_q = []
        for idx, vi in enumerate(tuple_i[:-1]):
            cur_q.append("{}=='{}'".format(sensi_atts[idx], vi))
        tuple_df = df.query(" and ".join(cur_q))
        metrics_all = compute_evaluation_metric_binary(list(tuple_df[target_name]), list(tuple_df["pred_"+target_name]), target_label_order)
        res_dict[key_tuple] = {x: round(metrics_all[x], round_digit) for x in metrics_all if x in output_metrics}

    return res_dict

def autolabel(rects, ax, font_size):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height), fontsize = font_size - 3,
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

def draw_bar_plot(f_label, y_name, focus_atts, fig_title, performance_flag=False, export_images = False, save_path = ''):

    primary_key = remove_list_dup([item[0] for item in f_label.keys()])
    names = remove_list_dup([item[1] for item in f_label.keys()])

    x = np.arange(len(primary_key))
    width = 0.15

    f_label_data = []
    for key in [(x, y) for x in primary_key for y in names]:
        if key in f_label.keys():
            f_label_data.append(f_label[key][y_name])
        else:
            f_label_data.append(np.nan)
    # for key, value in f_label.items():
    #     if key[1] in names:
    #         f_label_data.append(value[y_name])
    #     else:
    #         f_label_data.append(0.0)

    f_label_data = np.array(f_label_data).reshape(len(primary_key), int(len(f_label_data)/2)).T

    fig, ax = plt.subplots(figsize=(10,6), dpi=100)

    rects_pos = [ax.bar(x+i*width, y, width, alpha=0.7, edgecolor='white', label=name) for i, (name, y) in enumerate(zip(names, f_label_data))]

#     you can change the font size HERE
    font_size = 14

    ax.set_ylabel(y_name, fontsize = font_size)
    ax.set_title(fig_title, fontsize = font_size)
    ax.set_xticks(x+width*((len(f_label_data)-1)/2))

    ax.set_xticklabels(primary_key, fontsize = font_size)
    ax.legend(fontsize = font_size - 3)
    plt.grid()
    plt.ylim([0.5, 1.1])

    for rect in rects_pos:
        autolabel(rect, ax, font_size)

    fig.tight_layout()

#     save to png
    if export_images:
        if not os.path.exists(save_path+"/images"):
            os.mkdir(save_path+"/images")
        plt.savefig(save_path+"/images/"+fig_title+'_performance.png')

    plt.show()

def int_to_string(n):
    return n.to_bytes(math.ceil(n.bit_length() / 8), 'little').decode()

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

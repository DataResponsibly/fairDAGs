#!/usr/bin/env python
# coding: utf-8

# # Fairness-Aware Instrumentation of ML-Pipelines

# ## Preparations

import os
from collections import defaultdict
import inspect
import pandas as pd
import numpy as np
from scipy import stats
import re
from graphviz import Digraph
import pickle
import random
import plotly.express as px


from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize, KBinsDiscretizer, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from utils import *

import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_colwidth',1000)
np.set_printoptions(precision = 4)
pd.set_option("display.precision", 4)
pd.set_option('expand_frame_repr', True)


# ## Logs Part

# In[2]:


# Current Version
def describe_ver(pipeline_to_test, cat_col, numerical_col, sensi_atts, target_name, training, save_path):
    to_csv_count = 1
    log_dict = {}
    plot_dict = {}
    raw_func = inspect.getsource(pipeline_to_test)


    input_args, executable_list, outputs = func_aggregation(raw_func)

    for line in input_args:
        exec(line)

    prev = {}

    numerical_metric_list = ['count', 'missing_count', 'median', 'mad', 'range']
    numerical_df = pd.DataFrame(np.inf, index = numerical_col, columns = numerical_metric_list)

    cat_metric_list = ['missing_count', 'num_class', 'class_count', 'class_percent']
    cat_df = pd.DataFrame(np.inf, index = cat_col, columns = cat_metric_list)


    ######################################
    # Execution
    ######################################
    for cur_line in executable_list:
        print_bool = False
        exec(cur_line)
#         print(cur_line)
        if '#' in cur_line:
            continue
        try:
            if str(eval(f"type({cur_line.split('=')[0].strip()})")) == "<class 'pandas.core.frame.DataFrame'>":

                target_df = cur_line.split('=')[0].strip()

                plot_dict[line_cleansing(cur_line)] = static_label(eval(target_df), sensi_atts, target_name)
                eval(target_df).to_csv(save_path+'/checkpoints/csv/training/o'+str(to_csv_count)+'.csv' if training else save_path+'/checkpoints/csv/testing/o'+str(to_csv_count)+'.csv')
                to_csv_count+=1

                col_list = eval(target_df).columns.tolist()
                numerical_col_sub = [i for i in numerical_col if i in col_list]

                cat_col_sub = [j for j in cat_col if j in col_list]

                if len(numerical_col_sub) != 0:
                    ######################################################################################
                    # numerical features & metrices
                    # counts, missing values, Median and MAD, range/scaling
                    ######################################################################################
                    for numeric_feature in numerical_col_sub:

                        numerical_df = cal_numerical(eval(target_df), numeric_feature, numerical_df)

                if len(cat_col_sub) != 0:
                    ######################################################################################
                    # categorical features & metrices
                    # missing values, number of classes, counts for each group, percentage for each group
                    ######################################################################################
                    for cat_feature in cat_col_sub:

                        cat_df = cal_categorical(eval(target_df), cat_feature, cat_df)

                ######################################################################################
                # Comparison occurs here!
                ######################################################################################

                if len(prev) != 0:
                    if len(numerical_col_sub) != 0:
                        numerical_dif = numerical_df - prev['numerical']
                        if (numerical_dif.values != 0).any():
                            log_dict[line_cleansing(cur_line)] = {'num': numerical_dif}

                ##################################
                # ⬆️ numerical
                # ⬇️ categorical
                ##################################
                    if len(cat_col_sub) != 0:
                        cat_dif = get_categorical_dif(cat_df, cat_metric_list, prev['categorical'])
                        cat_dif_flag = check_cat_dif(cat_dif)
                        if cat_dif_flag:

                            log_dict[line_cleansing(cur_line)] = {'cat':cat_dif}

                print_bool = True


                # save the output for next round comparison
                prev['numerical'] = numerical_df.copy()
                prev['categorical'] = cat_df.copy()

            elif str(eval(f"type({cur_line.split('=')[0].strip()})")).startswith("<class 'sklearn"):
                pass
            else:
                pass

        except:
            if len(numerical_col_sub) != 0:
                ######################################################################################
                # numerical features & metrices
                # counts, missing values, Median and MAD, range/scaling
                ######################################################################################
                for numeric_feature in numerical_col:

                    numerical_df = cal_numerical(eval(target_df), numeric_feature, numerical_df)
            if len(cat_col_sub) != 0:
                ######################################################################################
                # categorical features & metrices
                # missing values, number of classes, counts for each group, percentage for each group
                ######################################################################################
                for cat_feature in cat_col:

                    cat_df = cal_categorical(eval(target_df), cat_feature, cat_df)

            ######################################################################################
            # Comparison occurs here!
            ######################################################################################
            if len(prev) != 0:
                if len(numerical_col_sub) != 0:
                    numerical_dif = numerical_df - prev['numerical']
                    if (numerical_dif.values != 0).any():
                        log_dict[line_cleansing(cur_line)] = {'num':numerical_dif}


            ##################################
            # ⬆️ numerical
            # ⬇️ categorical
            ##################################
                if len(cat_col_sub) != 0:
                    cat_dif = get_categorical_dif(cat_df, cat_metric_list, prev['categorical'])
                    cat_dif_flag = check_cat_dif(cat_dif)
                    if cat_dif_flag:

                        log_dict[line_cleansing(cur_line)] = {'cat':cat_dif}


            print_bool = True

            # save the output for next round comparison
            prev['numerical'] = numerical_df.copy()
            prev['categorical'] = cat_df.copy()


    nested_graph = pipeline_to_dataflow_graph(eval(f'{outputs[0]}'))

    # print('####################### Start Sklearn Pipeline #######################')

    for item in nested_graph:
        ######################################################################################
        # numerical features & metrices
        # counts, missing values, Median and MAD, range/scaling
        ######################################################################################
        if item.name in numerical_col:
            numeric_feature = item.name

            eval(target_df)[item.name] = item.operation.fit_transform(eval(target_df)[item.name].values.reshape(-1,1))
            # print('-------------------------------------------------------')
            # print(f"Operations {str(item.operation).split('(')[0]} on {item.name}")
            # print('-------------------------------------------------------')
            # print()
            plot_dict[line_cleansing(f"{item.name}__{str(item.operation).split('(')[0]}")] = static_label(eval(target_df), sensi_atts, target_name)
            eval(target_df).to_csv(save_path+'/checkpoints/csv/training/o'+str(to_csv_count)+'.csv' if training else save_path+'/checkpoints/csv/testing/o'+str(to_csv_count)+'.csv')

            to_csv_count+=1
            ##############################
            # Metrices Calculation
            ##############################
            numerical_df = cal_numerical(eval(target_df), numeric_feature, numerical_df)

            ##############################
            # Comparison
            ##############################
            numerical_dif = numerical_df - prev['numerical']

            if (numerical_dif.loc[numeric_feature,:].values != 0).any():
                # print(f'Metrics: {mat} changed in {col} with value {dif}')
                # print('*'*10)
                # print('Changes in numerical features!')
                # display(numerical_dif.loc[numeric_feature,:].to_frame())
                log_dict[line_cleansing(f"{item.name}__{str(item.operation).split('(')[0]}")] =  {'num':numerical_dif.loc[numeric_feature,:].to_frame().transpose()}
                # print('*'*10)
                # print()

        ######################################################################################
        # categorical features & metrices
        # missing values, number of classes, counts for each group, percentage for each group
        ######################################################################################
        elif item.name in cat_col:
            cat_feature = item.name
            ##############################
            try:
                eval(target_df)[item.name] = item.operation.fit_transform(eval(target_df)[item.name].values.reshape(-1,1)).toarray()
            except:
                eval(target_df)[item.name] = item.operation.fit_transform(eval(target_df)[item.name].values.reshape(-1,1))
            plot_dict[line_cleansing(f"{item.name}__{str(item.operation).split('(')[0]}")] = static_label(eval(target_df), sensi_atts, target_name)
            eval(target_df).to_csv(save_path+'/checkpoints/csv/training/o'+str(to_csv_count)+'.csv' if training else save_path+'/checkpoints/csv/testing/o'+str(to_csv_count)+'.csv')
            to_csv_count+=1

            ##############################
            # Metrices Calculation
            ##############################
            cat_df = cal_categorical(eval(target_df), cat_feature, cat_df)

            ##############################
            # Comparison
            ##############################
            cat_dif = get_categorical_dif(cat_df, cat_metric_list, prev['categorical'])
            cat_dif_flag = check_cat_dif(cat_dif)
            if cat_dif_flag:
                # print('*'*10)
                # print('Changes in categorical features!')
                # display(cat_dif.loc[cat_feature,:].to_frame())
                log_dict[line_cleansing(f"{item.name}__{str(item.operation).split('(')[0]}")] = {'cat':cat_dif.loc[cat_feature,:].to_frame().transpose()}
                # print('*'*10)

        else:
            try:
                eval(target_df)[item.name] = item.operation.fit_transform(eval(target_df)[item.name].values.reshape(-1,1)).toarray()
            except:
                pass
            plot_dict[line_cleansing(f"{item.name}__{str(item.operation).split('(')[0]}")] = static_label(eval(target_df), sensi_atts, target_name)
            eval(target_df).to_csv(save_path+'/checkpoints/csv/training/o'+str(to_csv_count)+'.csv' if training else save_path+'/checkpoints/csv/testing/o'+str(to_csv_count)+'.csv')
            to_csv_count+=1

        prev['numerical'] = numerical_df.copy()
        prev['categorical'] = cat_df.copy()

    # run classifier
    classi_match = re.findall("'classifier',.\w+\(\)", executable_list[-1])[0].split(', ')[-1]
    clf =  eval(classi_match)
    if training:
        to_train = eval(target_df).select_dtypes(include=['int', 'float64'])
        eval(target_df)[target_name] = eval('labels')

        clf.fit(to_train, eval('labels'))
        eval(target_df)['pred_'+target_name] = clf.predict(to_train)
    else:
        # to_train = eval(target_df).select_dtypes(exclude=['object'])
        eval(target_df)[target_name] = eval('labels')
        # clf.fit(to_train, eval('labels'))
        # eval(target_df)['pred_'+target_name] = clf.predict(to_train)
    # print(eval(target))
    return log_dict, plot_dict, eval(target_df), clf


# ## DAGs Part

# In[405]:

def find_pd_lines(pipeline_func):
    pipeline_func = inspect.getsource(pipeline_func)
    pd_lines = []
    input_args , executable_list, _ = func_aggregation(pipeline_func)
    for line in input_args:
        exec(line)
    for cur_line in executable_list:
        exec(cur_line)
        try:
            if 'inplace' in cur_line:
                pd_lines.append(cur_line)
            elif str(eval(f"type({cur_line.split('=')[0].strip()})")).startswith("<class 'pandas"):
                pd_lines.append(cur_line)
        except:
            pass
    return pd_lines

def pd_to_dataflow_graph(pipeline_func, log_list, parent_vertices=[]):
    executable_list = find_pd_lines(pipeline_func)
    graph = []
    previous = []

    for line in executable_list:
        line.replace('{','').replace('}', '')
        if 'inplace' in line and '#' not in line:
            log_list.append(line_cleansing(line))

            df_name = line.split('.')[0]
            func_name = line.split('.')[1].split('(')[0].strip()
            col_effect = line.split('[')[1].split(']')[0].strip()
            if len(previous) > 1:
                for node in previous:
                    if node.name == df_name:
                        vertex = DataFlowVertex([node], df_name+'_drop', func_name+' '+col_effect, col_effect)
                        previous.append(vertex)
                        previous.remove(node)
            else:
                vertex = DataFlowVertex(previous, df_name+'_drop', func_name+' '+col_effect, col_effect)
                previous = [vertex]
        else:
            var_name = line.split('=')[0].strip()

            # match ".func_name(...)"
            pd_func = re.search('\.\s*([_a-z]*)\s*\(',line)
            if pd_func:
                func_name = pd_func.group(1)
                params = re.search('\(([^\)]*)\)',line)  #"(...)"

                if params:
                    params = params.group(1).strip()

                    if func_name == 'read_csv': #df = pd.read_csv(path)
                        vertex = DataFlowVertex(parent_vertices,var_name, func_name, params)
                        previous.append(vertex)
                        log_list.append(line_cleansing(line))
                    elif func_name in ['join','merge','concat']:
                        log_list.append(line_cleansing(line))
                        if func_name == 'concat': #df_new = pd.concat([df1,df2],keys=[])
                            df_names = [item.strip() for item in params.split(']')[0].strip().strip('[]').split(',')]

                        else: # df_new = df1.join/merge(df2,on='...',how='...')
                            df_names = [line.split('=')[1].strip().split('.')[0], params.split(',')[0].strip()]
                        parent_vertices = search_vertex_by_names(df_names, graph) #search in graph by df_names
                        vertex = DataFlowVertex(parent_vertices, var_name, func_name, params) #TODO vertex name?
                        previous = [vertex] + list(set(previous) - set(parent_vertices))
                    elif 'lambda' in params:
                        log_list.append(line_cleansing(line))
                        cols = var_name.split('[')[1].split(']')[0].strip()
                        vertex = DataFlowVertex(previous, func_name+' '+cols, func_name, params)
                        previous = [vertex]
                    elif '[' in var_name:
                        log_list.append(line_cleansing(line))
                        cols = var_name.split('[')[1].split(']')[0].strip()
                        vertex = DataFlowVertex(previous, func_name+' '+cols+' '+params, func_name, params)
                        previous = [vertex]
                    else:
                        log_list.append(line_cleansing(line))
                        vertex = DataFlowVertex(previous, func_name+' '+params, func_name, params)
                        previous = [vertex]


            # filter operation: "df[[cols]]", "df[condition]","df.loc[]","df.iloc[]"
            else:
                if '[[' in line:
                    is_filter = re.search('\[([^\]]*)\]',line) #"[...]"
                else:
                    is_filter = re.search('\(([^\)]*)\)',line) #"[...]"
                if is_filter:
                    log_list.append(line_cleansing(line))
                    filter_cond = is_filter.group(1).strip('[').strip(']')
                    vertex = DataFlowVertex(previous, 'select '+filter_cond, 'filter', filter_cond)
                    previous = [vertex]

        graph.append(vertex)

    return graph, previous, log_list


def sklearn_to_dataflow_graph(pipeline, log_list, parent_vertices=[]):
    graph = pipeline_to_dataflow_graph_full(pipeline)
    graph_dict = pipeline_to_dataflow_graph(pipeline)
    for node in graph_dict:
        log_list.append(line_cleansing(f"{node.name}__{str(node.operation).split('(')[0]}"))
    for node in graph:
        if node.parent_vertices==[]:
            node.parent_vertices = parent_vertices
    return graph, log_list

def visualize(nested_graph, log_list, save_path, dag_save):
    no_nodes = len(log_list)
    rand_rgb = ['#191970', '#ff0000', '#006400', '#32cd32', '#ffd700', '#9932cc', '#ff69b4', '#8b4513', '#00ced1', '#d2691e'] if no_nodes <= 10 else ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(no_nodes)]
    dot = Digraph(comment='preprocessing_pipeline')
    dot.format = dag_save
    previous = {}
    for i, node in enumerate(nested_graph):
        node_name = node.name.replace('>=', '&ge;').replace('<=', '&le;')[:50] +' ...' if len(node.name)>40 else node.name.replace('>=', '&ge;').replace('<=', '&le;')
        dot.node(node.name, color = rand_rgb[i], fontcolor = rand_rgb[i], href = "{{url_for('home', type="+str(log_list[i])+")}}", label = f'''<<font POINT-SIZE="14"><b>{node_name}</b></font><br/><font POINT-SIZE="10">{node.operation}</font>>''')
        # dot.node(node.name, color = rand_rgb[i], fontcolor = rand_rgb[i], href = "{{url_for('home', type="+str(log_list[i])+")}}", label = f"{node.name}\n{node.operation}")
        parents = node.parent_vertices

        for parent in parents:
            dot.edge(parent.name, node.name)

    if not os.path.exists(save_path+'/DAG'):
        os.mkdir(save_path+'/DAG')
    dot.render(save_path+'/DAG/pipeline', view=False)
    return dot, rand_rgb


# ## Combine and make Func Wrapper


def tracer(cat_col, numerical_col, sensi_atts, target_name, training = True, save_path = '', dag_save = 'pdf'):
    def wrapper(func):
        def call(*args, **kwargs):
            if not os.path.exists('experiments'):
                os.mkdir('experiments')
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            if not os.path.exists(save_path+'/checkpoints'):
                os.mkdir(save_path+'/checkpoints')
            if not os.path.exists(save_path+'/checkpoints/csv'):
                os.mkdir(save_path+'/checkpoints/csv')
            if training and not os.path.exists(save_path+'/checkpoints/csv/training'):
                os.mkdir(save_path+'/checkpoints/csv/training')
            if not training and not os.path.exists(save_path+'/checkpoints/csv/testing'):
                os.mkdir(save_path+'/checkpoints/csv/testing')
            log_dict, plot_dict, target_df, clf = describe_ver(func, cat_col, numerical_col, sensi_atts, target_name, training, save_path)
            pickle.dump(clf, open(save_path+"/checkpoints/clf.p", "wb"))
            if training:
                pickle.dump(target_df, open(save_path+"/checkpoints/target_df_train.p", "wb"))
                pickle.dump(log_dict, open(save_path+"/checkpoints/log_dict_train.p", "wb"))
                pickle.dump(plot_dict, open(save_path+"/checkpoints/plot_dict_train.p", "wb"))
            else:
                pickle.dump(target_df, open(save_path+"/checkpoints/target_df_test.p", "wb"))
                pickle.dump(log_dict, open(save_path+"/checkpoints/log_dict_test.p", "wb"))
                pickle.dump(plot_dict, open(save_path+"/checkpoints/plot_dict_test.p", "wb"))
            log_list = []
            pd_graph, parent_vertices, log_list = pd_to_dataflow_graph(func, log_list)
            pipeline = func(*args, **kwargs)
            sklearn_graph, log_list = sklearn_to_dataflow_graph(pipeline, log_list, parent_vertices)
            pd_graph.extend(sklearn_graph)
            _, rand_rgb = visualize(pd_graph, log_list, save_path, dag_save)
            colors = dict(zip(log_list, rand_rgb))
            if training:
                pickle.dump(colors, open(save_path+"/checkpoints/rand_color_train.p", "wb"))
                pickle.dump(log_list, open(save_path+"/checkpoints/log_list_dag_train.p", "wb"))
            else:
                pickle.dump(colors, open(save_path+"/checkpoints/rand_color_test.p", "wb"))
                pickle.dump(log_list, open(save_path+"/checkpoints/log_list_dag_test.p", "wb"))
        return call
    return wrapper

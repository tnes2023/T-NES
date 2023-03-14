import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from collections import OrderedDict
import matplotlib.pyplot as plt
import seaborn as sns

def get_exp_result(result_path):
    print("**************************** print results ****************************")
    print(result_path)
    print("***********************************************************************")
    f = open(result_path, "r")
    result_file_name = result_path.split('/')[-1][:-4]
    try:
        dataset, model, samples_per_draw, method, solver, max_query, B, use_TD = result_file_name.split('_')
    except:
        try:
            samples_per_draw, method, max_query, B, use_TD, dataset = result_file_name.split('_')
            solver = method
        except:
            samples_per_draw, method, max_query, solver, use_TD = result_file_name.split('_')
            B = 1


    if method != 'zoo':
        exp_name = "{}_{}_{}_{}".format(method, samples_per_draw, B, use_TD)
    else:
        exp_name = "{}_{}_{}_{}".format(solver, samples_per_draw, B, use_TD)

    # try:
    #     batch_size, method, q_t, B, use_TD, dataset = result_file_name.split('_')
    #     has_dataset = True
    # except:
    #     batch_size, method, q_t, B, use_TD = result_file_name.split('_')
    #     has_dataset = False

    # q_t = int(q_t)
    lines = f.readlines()

    cer_dict = {}
    wer_dict = {}
    result_list = []
    result_query = []
    current_id = None
    for i in range(len(lines)):
        if "ID" in lines[i]:
            current_id = int(lines[i].split()[1])
            current_threshold = 0.1
            if current_id not in cer_dict.keys():
                cer_dict[current_id] = []
                wer_dict[current_id] = []
        elif "query =" in lines[i]:
            line_info = lines[i].split()
            query_time = int(line_info[2][:-1])
            if method == 'fd' or method == 'nes' or method == 'genetic':
                increase_cer = float(line_info[-4][:-1])
            else:
                increase_cer = float(line_info[-1])
            while increase_cer > current_threshold and current_threshold <= 0.6:
                cer_dict[current_id].append(query_time)
                current_threshold += 0.1

    # print(cer_dict)

    for i in range(1, 7):
        current_threshold = 0.1*i
        success_samples = 0
        total_queries = 0
        for id, queries in cer_dict.items():
            if len(queries) >= i:
                success_samples += 1
                total_queries += queries[i-1]
        if success_samples != 0:
            avg_queries = total_queries/success_samples
        else:
            avg_queries = -1
        print("Threshold: {}, success samples: {}, avg queries: {}".format(current_threshold, success_samples, avg_queries))
        # result_list.append("{}/{}({})".format(success_samples, len(cer_dict), avg_queries))
        result_list.append((success_samples/20, avg_queries))
        result_query.append(avg_queries)
    return exp_name, result_list, result_query

def get_exp_result2(result_path):
    print("**************************** print results ****************************")
    print(result_path)
    print("***********************************************************************")
    f = open(result_path, "r")
    result_file_name = result_path.split('/')[-1][:-4]
    try:
        dataset, model, samples_per_draw, method, solver, max_query, B, use_TD = result_file_name.split('_')
    except:
        try:
            samples_per_draw, method, max_query, B, use_TD, dataset = result_file_name.split('_')
            solver = method
        except:
            samples_per_draw, method, max_query, solver, use_TD = result_file_name.split('_')
            B = 1


    if method != 'zoo':
        exp_name = "{}_{}_{}_{}".format(method, samples_per_draw, B, use_TD)
    else:
        exp_name = "{}_{}_{}_{}".format(solver, samples_per_draw, B, use_TD)

    # try:
    #     batch_size, method, q_t, B, use_TD, dataset = result_file_name.split('_')
    #     has_dataset = True
    # except:
    #     batch_size, method, q_t, B, use_TD = result_file_name.split('_')
    #     has_dataset = False

    # q_t = int(q_t)
    lines = f.readlines()

    cer_dict = {}
    wer_dict = {}
    result_list = []
    result_query = []
    current_id = None
    for i in range(len(lines)):
        if "ID" in lines[i]:
            current_id = int(lines[i].split()[1])
            current_threshold = 0.1
            if current_id not in cer_dict.keys():
                cer_dict[current_id] = []
                wer_dict[current_id] = []
        elif "query =" in lines[i]:
            line_info = lines[i].split()
            query_time = int(line_info[2][:-1])
            if method == 'fd' or method == 'nes' or method == 'genetic':
                increase_cer = float(line_info[-4][:-1])
            else:
                increase_cer = float(line_info[-1])
            while increase_cer > current_threshold and current_threshold <= 0.6:
                cer_dict[current_id].append(query_time)
                current_threshold += 0.1

    # print(cer_dict)

    for i in range(1, 7):
        current_threshold = 0.1*i
        success_samples = 0
        total_queries = 0
        for id, queries in cer_dict.items():
            if len(queries) >= i:
                success_samples += 1
                total_queries += queries[i-1]
        if success_samples != 0:
            avg_queries = total_queries/success_samples
        else:
            avg_queries = -1
        print("Threshold: {}, success samples: {}, avg queries: {}".format(current_threshold, success_samples, avg_queries))
        # result_list.append("{}/{}({})".format(success_samples, len(cer_dict), avg_queries))
        result_list.append((success_samples/20))
        result_query.append(avg_queries)
    return exp_name, result_list, result_query

def exp(result_path):
    # "/home/tongch/Desktop/LibriSpeech_res/ds2_ted/nes/200_nes_500_0.01_False_ted.txt"
    print("**************************** print results ****************************")
    print("***********************************************************************")
    f = open(result_path, "r")
    result_file_name = result_path.split('/')[-1][:-4]
    try:
        batch_size, method, q_t, B, use_TD, dataset = result_file_name.split('_')
        has_dataset = True
    except:
        batch_size, method, q_t, B, use_TD = result_file_name.split('_')
        has_dataset = False

    q_t = int(q_t)
    lines = f.readlines()

    cer_dict = {}
    wer_dict = {}

    for i in range(len(lines)):
        j = 1
        if "ID" in lines[i]:
            id = int(lines[i].split()[1])
            if id not in cer_dict.keys():
                cer_dict[id] = []
                wer_dict[id] = []

    incre_CER = []
    wer = []
    for i in range(len(lines)):
        if "query times" in lines[i]:
            line_list = lines[i].split()
            incre_CER.append(format(float(line_list[-1]),'.5f'))
            # wer.append(format(float(line_list[-1]), '.5f'))

    id = 1

    for i in range(20):
        cer_dict[id] = incre_CER[q_t * i:q_t * (i + 1)]
        # wer_dict[id] = wer[q_t * i:q_t * (i + 1)]
        id += 1

    # print(cer_dict)

    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    for threshold in thresholds:
        a_1 = 0
        n = 0
        for ind in cer_dict.keys():
            a = []
            for i in range(len(cer_dict[ind])):
                a.append(round(float(cer_dict[ind][i]), 1))
            # print(len(a))
            try:
                a_1 += a.index(threshold)
                # print(ind,a_1)
            except ValueError:
                # print(ind)
                n += 1
                continue

        if (20-n) != 0:
            avg_q = a_1/(20-n)
        else:
            avg_q = 0

        if has_dataset:
            print("{}, {}, {}, {}, {}, {}".format(batch_size, method, q_t, B, use_TD, dataset))
            print("Threshold: {}, avg query: {}, success attack: {}".format(threshold, avg_q, 20-n))
            print("-----------------------------------------")
        else:
            print("{}, {}, {}, {}, {}".format(batch_size, method, q_t, B, use_TD))
            print("Threshold: {}, avg query: {}, success attack: {}".format(threshold, avg_q, 20-n))
            print("-----------------------------------------")


def result2addition():
    result, query = get_result("output_ted_ds2", "ted")
    df_result = pd.DataFrame(result, index=['0.1', '0.2', '0.3', '0.4', '0.5', '0.6'])
    df_query = pd.DataFrame(query, index=['0.1', '0.2', '0.3', '0.4', '0.5', '0.6'])

    groups = [
        ("fd_200_1.0_0", "fd_200_1.0_1"),
        ("nes_200_1.0_0", "nes_200_1.0_1"),
        ("genetic_200_1.0_0", "genetic_200_1.0_1"),
        ("adam_200_1.0_0", "adam_200_1.0_1"),
        ("newton_200_1.0_0", "newton_200_1.0_1"),
    ]

    for group in groups:
        diff = (df_result[group[1]] - df_result[group[0]])/df_result[group[1]]
        print(group[0])
        print(diff)

    print()

    for group in groups:
        diff = (df_query[group[1]] - df_query[group[0]])/df_query[group[1]]
        print(group[0])
        print(diff)

def get_result(path, dataset):
    result_files = os.listdir(path)
    result_files.sort()
    result = {}
    result_queries = {}
    for file in result_files:
        file_path = os.path.join(path, file)
        if file_path.split('.')[-1] == "txt" and dataset in file_path:
            print(file_path)
            # exp(file_path)
            exp_name, exp_result, exp_query = get_exp_result2(file_path)
            result[exp_name] = exp_result
            result_queries[exp_name] = exp_query
    result = OrderedDict(result)
    result_queries = OrderedDict(result_queries)

    return result, result_queries

def result2csv():
    path = 'output_new'
    datasets = ["ls"]

    for dataset in datasets:
        result_files = os.listdir(path)
        result_files.sort()
        result = {}
        result_queries = {}
        for file in result_files:
            file_path = os.path.join(path, file)
            if file_path.split('.')[-1] == "txt" and dataset in file_path:
                print(file_path)
                # exp(file_path)
                exp_name, exp_result, exp_query = get_exp_result(file_path)
                result[exp_name] = exp_result
                result_queries[exp_name] = exp_query
        result = OrderedDict(result)
        df = pd.DataFrame(result, index=['0.1', '0.2', '0.3', '0.4', '0.5', '0.6'])
        df.to_csv(dataset+'.csv')
        df = pd.DataFrame(result_queries, index=['0.1', '0.2', '0.3', '0.4', '0.5', '0.6'])
        df.to_csv(dataset+'_query.csv')


def query2plot(batch_size, ax):
    # sns.set_theme(style="whitegrid")
    # plt.figure(figsize=(8,6))

    x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    df = pd.read_csv("ls_query.csv")
    df['Increased Character Error Rates (ICERs)'] = x
    df = df[['Increased Character Error Rates (ICERs)', "nes_{}_1.0_1".format(batch_size), "fd_{}_1.0_0".format(batch_size), "genetic_{}_1.0_0".format(batch_size),
             "adam_{}_1.0_0".format(batch_size), "newton_{}_1.0_0".format(batch_size)]]
    df.columns = ['Increased Character Error Rates (ICERs)', 'T-NES', 'FD', 'GA', 'ZOO-Adam', 'ZOO-Newton']
    df = pd.melt(df, id_vars=['Increased Character Error Rates (ICERs)'], value_vars=['T-NES', 'FD', 'GA', 'ZOO-Adam', 'ZOO-Newton'],var_name='Method', value_name='Query cost')
    sns.lineplot(data=df, x='Increased Character Error Rates (ICERs)', y='Query cost', hue='Method', style='Method', markers=True, linewidth = 6, ax=ax)
    ax.set_title('{} coodinates in a batch'.format(batch_size))
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(22)
        # item.set_weight('bold')
        # item.set_fontweight('bold')

    ax.legend(ncol=2, fontsize=22)

    # sns.lineplot(x="increased_cer", y="nes_500_1.0_1", data=df)
    # sns.lineplot(x="increased_cer", y="fd_500_1.0_0", data=df)
    # sns.lineplot(x="increased_cer", y="genetic_500_1.0_0", data=df)
    # sns.lineplot(x="increased_cer", y="adam_500_1.0_0", data=df)
    # sns.lineplot(x="increased_cer", y="newton_500_1.0_0", data=df)
    # sns.lineplot(x="increased_cer", y="nes_500_1.0_1", data=df)
    # plt.plot(x, df["nes_500_1.0_1"], label="T-NES")
    # plt.plot(x, df["fd_500_1.0_0"], label="FD")
    # plt.plot(x, df["genetic_500_1.0_0"], label="Genetic")
    # plt.plot(x, df["adam_500_1.0_0"], label="ZOO-Adam")
    # plt.plot(x, df["newton_500_1.0_0"], label="ZOO-Newton")
    # # plt.xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    # plt.xlabel("Increased CER")
    # plt.ylabel("Iterations")
    # plt.legend()
    # plt.tight_layout()
    # plt.show()


result2csv()
# query2plot()
# result2csv()
# result2addition()
#
# sns.set()
# fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharex=True)
# query2plot(200, axes[0])
# query2plot(500, axes[1])
# plt.tight_layout()
# # plt.show()
# plt.savefig('ablation_batch.pdf')
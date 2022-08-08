from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.cm as cm
import matplotlib.patches as mpatches

import click
import sys
import yaml
import os
project_dir = os.path.dirname(os.path.join(os.getcwd(), __file__))
sys.path.append(os.path.normpath(os.path.join(project_dir, '..')))

from utils.constants import (
    TEMP_MODELS_PATH,
    KUBE_YAMLS_PATH
    )

memory_dictionary = {}
cpu_dictionary = {}


def save_data(dict, usage, name, version, bat):
    if not bat in dict.keys():
        dict[bat] = []
    dict[bat].append((name + version, max(usage)))



def file_reader_text(file, mode):
    with open(file) as f:
        for line in f:
            data = line.split()
            model_name = data[2]
            model_version = data[3]
            bat = data[6]
            usage = "".join(data[8:])
            usage = usage.replace("["," ")
            usage = usage.replace("]"," ")
            usage = usage.replace(","," ")
            usage = usage.lstrip()

            usage = list(map(float, usage.split()))
            if mode == 1:
                save_data(memory_dictionary, usage, model_name,model_version, bat)
                
            else:
                save_data(cpu_dictionary, usage, model_name,model_version, bat)

    if mode == 1:
        temp = memory_dictionary
    else:
        temp = cpu_dictionary
    ms = []
    use = []
    for k in temp.keys():
        data = temp[k]
        for d in data:
            ms.append(d[0])
            use.append(d[1])
        plt.bar(ms, use)
        if mode == 1:
            plt.savefig(f"{k}-memory.png")
        else:
            plt.savefig(f"{k}-cpu.png")

            

def plot_data_lat(csv):
    df = pd.read_csv(csv)
    df1 = df.groupby(["model-name", "model-version", "batch-size"])[["latency"]].mean()
    print(df1)
    data = df1.to_numpy()
    different_batches = [[] for _ in range(6)]
    for i, d in enumerate(data):
        different_batches[i%6].append(d[0])
    models = ["inception1","inception2", "resnet1", "resnet2", "resnet3", "xception1", "xception2"]
    for i, b in enumerate(different_batches):
        plt.bar(models, b)
        plt.title(f"{2**(i+1)} batch size")
        plt.savefig(f"{2**(i+1)}.png")

def all_lat_in_one(csv):
    df = pd.read_csv(csv)
    df1 = df.groupby(["model-name", "model-version", "batch-size"])[["latency"]].mean()
    data = df1.to_numpy()
    print(df1)
    different_batches = [[] for _ in range(6)]
    for i, d in enumerate(data):
        different_batches[i%6].append(d[0])
    width = 0.25
    N = 7
    ind = np.arange(N)
    for i, db in enumerate(different_batches):
        print(len(db))
        plt.bar(ind+width*i, db,
        width = width, 
        label=f'{(i + 1)**2}')
    
    plt.savefig("all.png")


def plot_infer_prom(txt, type):
    config_file = "model-load"
    mv_dict = []
    names = {}

    N = 3
    ind = np.arange(6)  # the x locations for the groups
    width = 0.27
    config_file_path = os.path.join(
        KUBE_YAMLS_PATH, f"{config_file}.yaml")

    with open(config_file_path, 'r') as cf:
        config = yaml.safe_load(cf)

    model_names = config['model_names']
    versions = config['versions']
    for i, m in enumerate(model_names):
        for v in versions[i]:

            mv_dict.append(v)

    for i, m in enumerate(model_names):
        names[m] = [[] for _ in range(len(versions[i]))]

    with open(txt) as f:
        for line in f:
            data = line.split()
            model = data[2]
            version = data[3]
            batch_size = data[6]
            batch_size = int(batch_size)
            if batch_size == 64:
                break
            if type == "infer":
                latency = float(data[8])
                latency = latency / (10**6)
            else:
                latency = data[8].replace("[","")
                latency = latency.replace("]","")
                latency = latency.replace(",","")
                latency = latency.replace(" ","")
                latency = float(latency)
            v = int(version) - 1
            names[model][v].append(latency)

    
    # model_version_batch = [[] for i in range(6)]
    # for m in mv_dict.keys():
    #     vs = mv_dict[m]
    #     for v in vs:
    #         for i, lat in enumerate(v):
    #             model_version_batch[i].append(lat)
    
    
    
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']   
    

    # data = {}
    # for i, b in enumerate(model_version_batch):
    #     data[f"batch {2**i}"] = b
   
    for j in range(5):
        plot_array = []
        for key in names.keys():
            versions = names[key]
            for v in versions:
                plot_array.append(v[j])
        cs = []
        lg = []
        for i, n in enumerate(names.keys()):
            data = names[n]
            for k in data:
                cs.append(colors[i])
            lg.append([colors[i], n])
        plot_array1, new_mv_dict = zip(*sorted(zip(plot_array, mv_dict)))
        plot_array, cs = zip(*sorted(zip(plot_array, cs)))

        n_bars = sum(len(l) for l in versions)
        fig, ax = plt.subplots(figsize=(8, 6))
        # The width of a single bar
        bar_width = 0.3

        # # List containing handles for the drawn bars, used for the legend
        bars = set()
        x_offset = 0
        c = 0
        xs = []
        
        
        # # Iterate over all data
        for i, (name, values) in enumerate(names.items()):
            # The offset in x direction of that bar

        #     # Draw a bar for every value of that type
            for x, y in enumerate(values):
                bar = ax.bar(x/3 + x_offset, plot_array[c], width=bar_width * 0.9, color=cs[c])
                c += 1
                xs.append(x/3 + x_offset)
                bars.add(bar[0])

            x_offset += len(values)/3

        #     # Add a handle to the last drawn bar, which we'll need for the legend
        #         bars.append(bar[0])

        # # Draw legend if we need
        if True:

            patches = []
            for l in lg:
                patches.append(mpatches.Patch(color=l[0], label=l[1]))
            plt.legend(handles=patches,bbox_to_anchor=(1.04,1), loc="upper left" )

            if type == "infer":
                ax.set_ylabel(f"{type} time latency (seconds)")
            else:
                ax.set_ylabel(f"{type} usage")
            ax.set_xticks(xs, new_mv_dict)
            ax.tick_params(axis='both', which='minor', labelsize=1)
            plt.xticks(rotation=45)
        plt.savefig(f"big-run/experimental/v3/{type}-batchsize-{2**(j+1)}.png", bbox_inches="tight")


    # plt.savefig(f"big-run/ultimate/{type}.png")


def plot_per_batch(txt, type):
    config_file = "model-load"
    mv_dict = []
    names = {}

    N = 3
    ind = np.arange(6)  # the x locations for the groups
    width = 0.27
    config_file_path = os.path.join(
        KUBE_YAMLS_PATH, f"{config_file}.yaml")

    with open(config_file_path, 'r') as cf:
        config = yaml.safe_load(cf)

    model_names = config['model_names']
    versions = config['versions']
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']   

    for i, m in enumerate(model_names):
        for v in versions[i]:

            mv_dict.append(m+v)

    for i, m in enumerate(model_names):
        names[m] = [[] for _ in range(len(versions[i]))]

    with open(txt) as f:
        for line in f:
            data = line.split()
            model = data[2]
            version = data[3]
            batch_size = data[6]
            batch_size = int(batch_size)
            if batch_size == 64:
                break
            if type == "infer":
                latency = float(data[8])
                latency = latency / (10**6)
            else:
                latency = data[8].replace("[","")
                latency = latency.replace("]","")
                latency = latency.replace(",","")
                latency = latency.replace(" ","")
                latency = float(latency)
            v = int(version) - 1
            names[model][v].append(latency)

    c = 0

    for i, name in enumerate(names.keys()):
        # The offset in x direction of that bar
        values = names[name]
        print(name, values)
        return
    #     # Draw a bar for every value of that type
        bars = []
        for k, val in enumerate(values):
            fig, ax = plt.subplots(figsize=(8, 6))
            x_offset = 0
            xs = []
            for x, y in enumerate(val):
                bar = ax.bar(x/3 + x_offset, y, width=0.3 * 0.9, color=colors[x % len(colors)])
                bars.append(bar[0])
                xs.append(x/3 + x_offset)
            c += 1
            if True:

                
                plt.legend(bars,[f"batch {2**x}" for x in[1,2,3,4,5] ],bbox_to_anchor=(1.04,1), loc="upper left" )

                if type == "infer":
                    ax.set_ylabel(f"{type} time latency (seconds)")
                else:
                    ax.set_ylabel(f"{type} usage")
                ax.set_xticks(xs, [2, 4, 8, 16, 32])
                ax.tick_params(axis='both', which='minor', labelsize=1)
                plt.xticks(rotation=45)
            plt.savefig(f"big-run/experimental/v4/{type}-model-{mv_dict[c-1]}.png", bbox_inches="tight")
        


def plot_load_time(txt):
    model_names = []
    load_times = []
    with open(txt) as f:
        for line in f:
            data = line.split()
            model_name = data[3]
            load_time = float(data[5])
            model_names.append(model_name)
            load_times.append(load_time)
    plt.bar(model_names, load_times)
    plt.savefig("loadlatency.png")

where = "experimental"

plot_per_batch(f"big-run/{where}/infer-prom.txt","infer")
plot_per_batch(f"big-run/{where}/cpu.txt","cpu")
plot_per_batch(f"big-run/{where}/memory.txt","memory")





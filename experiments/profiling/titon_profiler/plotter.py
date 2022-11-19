from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.patches as mpatches
import glob
import sys
import yaml
import matplotlib.lines as mlines


import os
project_dir = os.path.dirname(os.path.join(os.getcwd(), __file__))
sys.path.append(os.path.normpath(os.path.join(project_dir, '../..')))

from utils.constants import KUBE_YAMLS_PATH

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
                bar = ax.bar(x/3 + x_offset, y, width=bar_width * 0.9, color=colors[i%len(colors)])
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
        plt.savefig(f"profile-exp4/1/images/permodel/regular/{type}-batchsize-{2**(j+1)}.png", bbox_inches="tight")


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
            print("hiiiii")
            plt.savefig(f"profile-exp4/1/images/perbatch/{type}-model-{mv_dict[c-1]}.png", bbox_inches="tight")
        
            

def plot_load_time(txt):
    model_names = []
    load_times = []
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
            model_name = data[3]
            load_time = float(data[5])
            model_names.append(model_name)
            load_times.append(load_time)
    plt.bar(model_names, load_times)
    plt.savefig("loadlatency.png")


import plotly.express as px
import plotly.io as pio

w,h = 1000,800

def show_all(df, title, size, batch_size):
    fig = px.scatter(df, width=w, height=h, title=title,
        x='time',  y='accuracy', log_x=True, color='model', hover_name='model')
    fig.update_traces(marker=dict(size=20))

    pio.write_image(fig, f"profile-exp4/1/images/acc-lat/{title}/{batch_size}.png")


def plot_accuracy_latency(txt, type, size):
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
    latencies = []
    with open(txt) as f:
        for line in f:
            data = line.split()
            model = data[2]
            version = data[3]
            batch_size = data[6]
            if int(batch_size) != size:
                continue
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
            latencies.append(latency)
            if type == "cpu" and batch_size == 32:
                latencies.append(414.687180389179591)

    
    df = pd.read_csv("results-imagenet.csv")
    df = df[df['model'].isin(mv_dict)][["model", "top1"]]
    pl_arr = []
    accs = []
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']   

    for i, mv in enumerate(mv_dict):
        print(mv, end = " ")
        print(latencies[i])
        pl_arr.append([mv, df.loc[df['model']==mv].iloc[0]['top1'], latencies[i], colors[i%len(colors)]])
    
    final_df = pd.DataFrame(columns=['model', 'accuracy', f'time', 'type'])
    for i, m, v, t in pl_arr:
        final_df.loc[len(final_df)] = [i, m, v, t]
    show_all(final_df, f"{type}", "img-size", size)

def convert_to_float(data):
    data = data.replace("[","")
    data = data.replace("]","")

    data = data.replace(',',"")
    data = list(map(float, data.split()))
    return data

def create_file_on_doesnot_exists(root, where, per, kind):
    path = os.path.join(root, where, "images", per, kind)
    print(path)
    isExist = os.path.exists(path)

    if not isExist:
    
    # Create a new directory because it does not exist 
        os.makedirs(path)
        print("The new directory is created!")
    return path

def plot_array_data(txt, type, size, yaml_file, per, kind):
    config_file = yaml_file
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
           
            metric = convert_to_float(" ".join(data[8:]))
            if type == "memory":
                met = max(metric) - min(metric)
            elif type == 'cpu':
                met = max(metric) - min(metric)
            else:
                met = metric[-1]
            v = int(version)-1
            names[model][v].append(met)
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
                bar = ax.bar(x/3 + x_offset, y, width=bar_width * 0.9, color=colors[i%len(colors)])
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
        files = txt.split("/")
        path = create_file_on_doesnot_exists(files[0], files[1], per, kind)
        plt.savefig(f"{path}/{type}-batchsize-{2**(j+1)}.png", bbox_inches="tight")

def create_color_column(names):
    families = []
    for name in names:
        families.append(name.split("-")[0])
    colors = {}
    number = 12
    cmap = plt.get_cmap('gnuplot')
    all_colors = [cmap(i) for i in np.linspace(0, 1, number)]
    index = -1
    for fam in families:
        if fam not in colors:
            index += 1
            colors[fam] = [all_colors[index]]
        else:
            colors[fam].append(all_colors[index])
    all_returns = []
    for k in colors.keys():
        data = colors[k]
        for d in data:
            all_returns.append(d)
    return all_returns


all_models = {
    "resnet": 6,
    "xception": 2,
    "inception": 2,
    "visformer": 1,
    "regnetx":3,
    "vgg": 3,
    "beit":2,
    "densenet": 1,
    "vit": 1,
    "legacy": 1
}
def plot_perf_analyzer(path):
    csv_files = glob.glob(os.path.join(path, "*.csv"))
    frames = []
    ret = []
    for f in csv_files:
        df = pd.read_csv(f)
        name = f.split("\\")[-1].split(".")[0].split("/")[-1]
        df["name"] = name
        group = name.split("-")[0]
        df["group"] = group
        frames.append(df)
        for m in all_models.keys():
            if m in group:
                    family = m
        ret.append([group, df.iloc[0]["Inferences/Second"], "thoughput",2, family ])
    return ret
        # plt.legend(handles=patches,bbox_to_anchor=(1.01,1), loc="upper left" )
        # plt.xticks(rotation = 90)
        # plt.ylabel(f'{val}({y_axis[i]})', size = 30)
        # plt.xlabel("model-version")
        # pa = create_file_on_doesnot_exists(path.split("/")[0], path.split("/")[1], "perf", "perf")
        # plt.savefig(f"{pa}/{data_to_plots[i]}.png")


all_models = {
     "yolov5s" :1,
     "yolov5n6": 1,
     "yolov5m6": 1,
     "yolov5s6": 1,
     "yolov5l6": 1,
     "yolov5x6": 1
}
number = 12
cmap = plt.get_cmap('gnuplot')
all_colors = [cmap(i) for i in np.linspace(0, 1, number)]
for i, k in enumerate(all_models.keys()):
    all_models[k] = all_colors[i]

def plot_all(txt, type):
    ret = []
    with open(txt) as f:
        for line in f:
            data = line.split()
            model = data[2]
            batch_size = data[6]
            batch_size = int(batch_size)
           
            metric = convert_to_float(" ".join(data[8:]))
            if type == "memory":
                met = max(metric)
            elif type == 'cpu':
                met = max(metric) - min(metric)
            else:
                met = metric[-1]
            for m in all_models.keys():
                if m in model:
                    family = m
            ret.append([model, met, type, batch_size, family])
    return ret

root = "profile-exp6-cores"
where = "8-new-batch"
f_names = ["infer-prom.txt", "cpu.txt", "memory.txt"]
types = ["infer", "cpu", "memory"]
yaml_file = "model-load"
per = "models"
kind = "regular"

def batchy_plot():
    df = []
    plt.rcParams["figure.figsize"] = (16,12)

    global f_names
    f_names = f_names[:2]
    for i, f in enumerate(f_names):
        df.append(plot_all(f"{root}/{where}/{f}", types[i]))

    infer_data = df[0]
    xs = []
    x_offset = 0
    bar_width = 0.3
    all_colors = [cmap(i) for i in np.linspace(0, 1, 5)]
    x_labels = [f"batch {2 ** (i + 1)}" for i in range(5)]
    print(len(infer_data))
    for x, data in enumerate(infer_data):
        plt.bar(x/3 + x_offset, data[1]/(10**6*20),label=data[3], width=bar_width * 0.9, color=all_colors[x])
        xs.append(x/3 + x_offset)
    
    cpu_data = df[1]
    y = []
    for d in cpu_data:
        y.append(d[1]/170)
    plt.plot(xs, y, color='blue', marker='o', linestyle='dashed',
     linewidth=2, markersize=12)
    patches = []
    for i, k in enumerate(all_colors):
        patches.append(mpatches.Patch(color=all_colors[i], label=x_labels[i]))
    patches.append( mlines.Line2D([], [], color='blue', marker='.',
                          markersize=15, label='cpu(seconds \n usage total)'))
    plt.xticks(xs, x_labels)
    plt.legend(handles=patches,bbox_to_anchor=(1.00, 1), loc="upper left", prop={'size': 13} )
    plt.ylabel("time spend(s)", fontsize=24)
    title = 'compare inference time and cpu time of different batches of resnet 101'
    plt.title(title,fontsize = 24)
    plt.savefig("here2.png",bbox_inches='tight')
def four_plot_in_one():
    df = []
    for i, f in enumerate(f_names):
        df.append(plot_all(f"{root}/{where}/{f}", types[i]))
    df.append(plot_perf_analyzer(f"{root}/{where}"))

    figure, ax = plt.subplots(1, 2, figsize=(17, 15))
    infer_data = df[0]
    infer_data.sort(key=lambda i: i[1])
    # infer_data = infer_data.sort(key=lambda i: i[1])
    x_offset = 0
    bar_width = 0.3
    labels= []
    ind = []
    for x, data in enumerate(infer_data):
        bar = ax[0,0].bar(x/3 + x_offset, data[1]/(10**6),label=data[0], width=bar_width * 0.9, color=all_models[data[-1]])
        labels.append(data[0])
        ind.append(x/3 + x_offset)
    ax[0, 0].set_xticks(ind)
    ax[0, 0].set_xticklabels(labels, rotation = 90)
    ax[0,0].set_ylabel(f"inference time(s)", fontsize=18)
    labels= []
    ind = []
    cpu_data = df[1]
    cpu_data.sort(key=lambda i: i[1])
    labels = []
    for x, data in enumerate(cpu_data):
        bar = ax[0,1].bar(x/3 + x_offset, data[1],label=data[0], width=bar_width * 0.9, color=all_models[data[-1]])
        labels.append(data[0])
        ind.append(x/3 + x_offset)
    ax[0,1].set_ylabel(f"cpu usage time(s)", fontsize=18)

    ax[0,1].set_xticks(ind)
    ax[0,1].set_xticklabels(labels, rotation = 90)
    labels= []
    ind = []
    memory_data = df[2]
    memory_data.sort(key=lambda i: i[1])

    for x, data in enumerate(memory_data):
        bar = ax[1,0].bar(x/3 + x_offset, data[1],label=data[0], width=bar_width * 0.9, color=all_models[data[-1]])
        labels.append(data[0])
        ind.append(x/3 + x_offset)
    
    ax[1,0].set_ylabel(f"memory usage(bytes)", fontsize=18)

    ax[1, 0].set_xticks(ind)
    ax[1, 0].set_xticklabels(labels, rotation = 90)
    labels= []
    ind = []
    throuput_data = df[3]
    throuput_data.sort(key=lambda i: i[1])
    for x, data in enumerate(throuput_data):
        bar = ax[1,1].bar(x/3 + x_offset, data[1],label=data[0], width=bar_width * 0.9, color=all_models[data[-1]])
        labels.append(data[0])
        ind.append(x/3 + x_offset)
    ax[1, 1].set_xticks(ind)
    ax[1, 1].set_xticklabels(labels, rotation = 90)
    ax[1, 1].set_ylabel(f"throughput(Inference/seconds)", fontsize=18)
    patches = []
    for l in all_models.keys():
        patches.append(mpatches.Patch(color=all_models[l], label=l))
    plt.legend(handles=patches,bbox_to_anchor=(1.00,2.22), loc="upper left", prop={'size': 13} )
    figure.savefig("here.png")
    # print(df)
    



root = "../profile-exp7-object"
where = "1"
f_names = ["infer-prom.txt", "cpu.txt", "memory.txt"]
types = ["infer", "cpu", "memory"]
yaml_file = "model-load"
per = "models"
kind = "regular"
# for k, f in enumerate(f_names):
#     for i in [2]:
#         plot_array_data(f"{root}/{where}/{f}",types[k], i, yaml_file, per, kind)

# plot_perf_analyzer(f"{root}/{where}")
batchy_plot()
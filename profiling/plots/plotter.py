from matplotlib import pyplot as plt
import pandas as pd
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
    data = df1.to_numpy()
    different_batches = [[] for _ in range(6)]
    for i, d in enumerate(data):
        different_batches[i%6].append(d[0])
    models = ["inception1","inception2", "resnet1", "resnet2", "resnet3", "xception1", "xception2"]
    for i, b in enumerate(different_batches):
        plt.bar(models, b)
        plt.title(f"{2**(i+1)} batch size")
        plt.savefig(f"{2**(i+1)}.png")


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



file_reader_text("../memory.txt", 1)
file_reader_text("../cpu.txt", 2)

print(memory_dictionary)
print(cpu_dictionary)


import os, tarfile
import bz2
import json
from builder_req import build_workload

def create_workload(day = '25'):
    tar = tarfile.open("twitter-2018-04-25.tar")
    days = set()
    print(len([i for i in tar.getmembers() if i.name.endswith("bz2")]))
    my_test_day = day
    for file in tar.getmembers():
        if file.name.endswith("bz2"):
            day = file.name.split("/")[2]
            if day == my_test_day:
                print(file.name)
                days.add(file)

    time_series = {}
    print(len(days))
    for member in days:
        f=tar.extractfile(member)
        with bz2.open(f, "rt") as bzinput:
            for i, line in enumerate(bzinput):
                tweets = json.loads(line)
                if len(tweets)> 1:
                    json_object = tweets

                    if 'created_at' in json_object:
                        if json_object['created_at'] in time_series.keys():
                            time_series[json_object['created_at']]+=1
                        else:
                            time_series[json_object['created_at']] = 1

    print("reading complete")
    print()
    print(time_series.keys().__len__())

    with open('data.json', 'w') as fp:
        json.dump(time_series, fp,  indent=4)
    build_workload()

create_workload()
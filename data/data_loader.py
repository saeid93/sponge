import os, tarfile
import bz2
import json

tar = tarfile.open("twitter-2018-04-25.tar")
days = set()
print(len([i for i in tar.getmembers() if i.name.endswith("bz2")]))
my_test_day = '25'
for file in tar.getmembers():
    if file.name.endswith("bz2"):
        day = file.name.split("/")[2]
        if day == my_test_day:
            days.add(file)

time_series = {}
for member in days:
    f=tar.extractfile(member)
    print(f.name)
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



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
df = pd.read_csv("data/data2.csv", index_col="time")
time_series = df["apparent"]


def data_slicing(raw_data, nr_samples=1440):
    training_examples = []
    for i in range(len(raw_data)//nr_samples):
        training_examples.append(raw_data.values[i*nr_samples:nr_samples + i*nr_samples])

    return training_examples


def data_compression(raw_data, nr_samples=1440):
    training_examples = []
    for i in range(len(raw_data)//nr_samples):
        training_examples.append(sum(raw_data.values[i*nr_samples:nr_samples + i*nr_samples]) >= nr_samples//2)

    return training_examples


def data_compression_1d(raw_data, nr_samples=1440):
    training_examples = []
    for i in range(len(raw_data)//nr_samples):
        training_examples.append(True if sum(raw_data.values[i*nr_samples:nr_samples + i*nr_samples]) >= nr_samples//2 else False)

    return training_examples


import os
import random
directory = 'training_data'

training_data_raw = {}
for dirname in os.listdir(directory):
    d = os.path.join(directory, dirname)
    for filename in os.listdir(d):
        f = os.path.join(d, filename)
        f_handle = open(f, "r")
        data = f_handle.read()
        f_handle.close()

        data = data.split(",")
        res = []
        for i in range(len(data)):
            try:
                res.append(float(data[i]))
            except:
                pass
        training_data_raw[filename] = res


hairdryer_on = [training_data_raw["hairdryer_on_1.csv"],
                training_data_raw["hairdryer_on_2.csv"],
                training_data_raw["hairdryer_on_3.csv"],
                training_data_raw["hairdryer_on_4.csv"],
                training_data_raw["hairdryer_on_5.csv"]]

hairdryer_off = [training_data_raw["hairdryer_off_1.csv"],
                 training_data_raw["hairdryer_off_2.csv"],
                 training_data_raw["hairdryer_off_3.csv"],
                 training_data_raw["hairdryer_off_4.csv"],
                 training_data_raw["hairdryer_off_5.csv"]]


kettle_on = [training_data_raw["kettle_on_1.csv"],
                training_data_raw["kettle_on_2.csv"],
                training_data_raw["kettle_on_3.csv"],
                training_data_raw["kettle_on_4.csv"],
                training_data_raw["kettle_on_5.csv"]]

kettle_off = [training_data_raw["kettle_off_1.csv"],
                 training_data_raw["kettle_off_2.csv"],
                 training_data_raw["kettle_off_3.csv"],
                 training_data_raw["kettle_off_4.csv"],
                 training_data_raw["kettle_off_5.csv"]]

vacuum_on = [training_data_raw["vacuum_on_1.csv"],
                training_data_raw["vacuum_on_2.csv"],
                training_data_raw["vacuum_on_3.csv"],
                training_data_raw["vacuum_on_4.csv"],
                training_data_raw["vacuum_on_5.csv"]]

vacuum_off = [training_data_raw["vacuum_off_1.csv"],
                 training_data_raw["vacuum_off_2.csv"],
                 training_data_raw["vacuum_off_3.csv"],
                 training_data_raw["vacuum_off_4.csv"],
                 training_data_raw["vacuum_off_5.csv"]]

lamp_on = [training_data_raw["lamp_on_1.csv"],
                training_data_raw["lamp_on_2.csv"],
                training_data_raw["lamp_on_3.csv"],
                training_data_raw["lamp_on_4.csv"],
                training_data_raw["lamp_on_5.csv"]]

lamp_off = [training_data_raw["lamp_off_1.csv"],
                 training_data_raw["lamp_off_2.csv"],
                 training_data_raw["lamp_off_3.csv"],
                 training_data_raw["lamp_off_4.csv"],
                 training_data_raw["lamp_off_5.csv"]]

fan_on = [training_data_raw["fan_on_1.csv"],
                training_data_raw["fan_on_2.csv"],
                training_data_raw["fan_on_3.csv"],
                training_data_raw["fan_on_4.csv"],
                training_data_raw["fan_on_5.csv"]]

fan_off = [training_data_raw["fan_off_1.csv"],
                 training_data_raw["fan_off_2.csv"],
                 training_data_raw["fan_off_3.csv"],
                 training_data_raw["fan_off_4.csv"],
                 training_data_raw["fan_off_5.csv"]]

on_off_data = [[fan_on, hairdryer_on, lamp_on, vacuum_on, kettle_on],
              [fan_off, hairdryer_off, lamp_off, vacuum_off, kettle_off]]


random.seed(4711)
training_data = []

current_devices = [0,0,0,0,0]
current_watt = [0,0,0,0,0]
current_time_series = []
devices_index = -1

for i in range(24*60*60):
    r = random.randint(0,100)
    point = []
    if len(current_time_series) == 0 and r > 95:
        device_index = random.randint(0,4)
        if current_devices[device_index] == 1 or r > 97:
            current_time_series = (on_off_data[current_devices[device_index]][device_index][random.randint(0,4)]).copy()
            current_devices[device_index] = 1-current_devices[device_index] # 1 to 0 or 0 to 1

    if len(current_time_series) > 0:
        current_watt[device_index] = current_time_series[0]
        current_time_series.pop(0)

    point.append(sum(current_watt))
    for num in current_devices: point.append(num)
    training_data.append(point)


f = open("training_data.csv", "w")
f.write("P,hairdryer,kettle,vacuum,lamp,fan\n")
for point in training_data:
    f.write(str(point[0]))
    f.write(",")
    f.write(str(point[1]))
    f.write(",")
    f.write(str(point[2]))
    f.write(",")
    f.write(str(point[3]))
    f.write(",")
    f.write(str(point[4]))
    f.write(",")
    f.write(str(point[5]))
    f.write("\n")
f.close()



import time
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import matthews_corrcoef

#X_train, X_test, y_train, y_test = train_test_split(X, y[:730], test_size=0.3, random_state=42, shuffle=True)

training_data = pd.read_csv("training_data.csv")
X_train = data_slicing(training_data["P"],60)
y_train = data_compression(training_data.iloc[:, 1:6],60)

y_train_x = [0,0,0,0,0]
y_train_x[0] = data_compression_1d(training_data.iloc[:, 1:2],60)
y_train_x[1] = data_compression_1d(training_data.iloc[:, 2:3],60)
y_train_x[2] = data_compression_1d(training_data.iloc[:, 3:4],60)
y_train_x[3] = data_compression_1d(training_data.iloc[:, 4:5],60)
y_train_x[4] = data_compression_1d(training_data.iloc[:, 5:6],60)



test_data = pd.read_csv("data/data2.csv", index_col="time")
X_test = data_slicing(test_data["apparent"], 60)
y_test = [[False,False,False,False,False],[False,False,False,True,False],[True,False,False,True,False],[True,False,False,True,False],[True,False,True,True,False],[True,False,True,True,False],[False,False,True,True,False],[False,False,True,True,False],[False,False,True,False,False],[False,False,False,False,False],[False,False,False,False,False],[True,False,False,False,False],[True,False,False,False,False],[True,True,False,False,False],[False,True,False,False,False],[False,True,False,False,False],[False,True,False,False,False],[False,False,False,False,False],[False,False,False,False,False],[False,False,False,False,False],[False,False,False,False,False],[False,False,False,False,True],[True,False,False,False,True],[True,False,False,False,True],[True,False,False,True,True],[False,False,False,True,True],[False,False,False,True,False],[False,False,False,True,False],[False,False,False,True,False],[False,False,False,False,False]]

y_test_x = [0,0,0,0,0]
y_test_x[0] = [row[0] for row in y_test]
y_test_x[1] = [row[1] for row in y_test]
y_test_x[2] = [row[2] for row in y_test]
y_test_x[3] = [row[3] for row in y_test]
y_test_x[4] = [row[4] for row in y_test]



print("-----------------------")
for i in range(5):
    t = time.time()
    clf = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train_x[i])
    print("learning time: " + str(time.time()-t))
    t = time.time()
    print("score: " + str(clf.score(X_test, y_test_x[i])))
    print("evaluation time: " + str(time.time()-t))
    print("")

print("-----------------------")
t = time.time()
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10, 10, 2), random_state=1).fit(X_train, y_train)
print("learning time: " + str(time.time()-t))
t = time.time()
print("score: " + str(clf.score(X_test, y_test)))
print("evaluation time: " + str(time.time()-t))
print("")

print("-----------------------")
for i in range(5):
    t = time.time()
    clf = AdaBoostClassifier(n_estimators=100, random_state=0).fit(X_train, y_train_x[i])
    print("learning time: " + str(time.time()-t))
    t = time.time()
    print("score: " + str(clf.score(X_test, y_test_x[i])))
    print("evaluation time: " + str(time.time()-t))
    print("")

print("-----------------------")
for i in range(5):
    t = time.time()
    clf = RandomForestClassifier(max_depth=10, n_estimators=50, random_state=0).fit(X_train, y_train_x[i])
    print("learning time: " + str(time.time()-t))
    t = time.time()
    print("score: " + str(clf.score(X_test, y_test_x[i])))
    print("evaluation time: " + str(time.time()-t))
    print("")

print("-----------------------")
for i in range(5):
    t = time.time()
    clf = LogisticRegression(random_state=0, max_iter=200000).fit(X_train, y_train_x[i])
    print("learning time: " + str(time.time()-t))
    t = time.time()
    print("score: " + str(clf.score(X_test, y_test_x[i])))
    print("evaluation time: " + str(time.time()-t))
    print("")


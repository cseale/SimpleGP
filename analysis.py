import fnmatch
import os
import numpy as np

pattern = '*maxtime20*cr1*resetTrue*.txt'
all_files = os.listdir('experiments/')
to_read_files = []
for name in all_files:
    to_read_files = ["experiments/" + file for file in all_files if fnmatch.fnmatch(file, pattern)]

evals = []
train_MSE = []
test_MSE = []
for name in to_read_files:
    f = open(name)
    prevline = "hi_hi_hi_hi"
    for i,x in enumerate(f):
        print(prevline)
        t = prevline
        if "Training" in x.split():
            train_MSE.append(float(f.readline()[5:-1]))
        if "Test:" in x.split():
            test_MSE.append(float(f.readline()[5:-1]))
        if "Function" in x.split():
            print("prev:{}".format(t))
            evals.append(int(prevline.split("_")[2]))
        prevline = x
        print(prevline)

train_MSE = np.array(train_MSE)
test_MSE = np.array(test_MSE)
print("pat: {} \nM_train: {} S_train: {}\nM_test: {} S_test:{}".format(pattern, np.mean(train_MSE), np.std(train_MSE), np.mean(test_MSE), np.std(test_MSE)))

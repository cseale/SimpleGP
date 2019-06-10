from statistics import mean, stdev
import os
import progressbar 
import re

def aggregateParams(experiment_dir, valType = "test_mse", printNum = 20, 
                    allStats = False):
    '''
    valType: which value to aggregate over: "test_mse", "numGen", "tree_size", 
            "train_mse", "diff_mse"
    printNum: number of mean mses to print
    allStats: whether to return also the value and file name of each single run
    '''
    files = os.listdir(experiment_dir)
    
    val_to_params = {}
    params_to_val = {}
    all_fileValues = []
    all_files = []
    
    # aggregate value
    with progressbar.ProgressBar(max_value=len(files)) as bar:
        for i, f in enumerate(files):
            key = get_key(f)
            d = file_read_from_tail(experiment_dir + f, 11)
            if d == None:
                continue
            if valType == "numGen": # get number of generations
                val = parse_numGen(d[0])
            elif valType == "test_mse": # get test MSE
                val = parse_mse(d[-3])
            elif valType == "tree_size": # get final tree size
                val = parse_treeSize(d[2])
            elif valType == "train_mse": # get final training MSE
                val = parse_mse(d[-6])
            elif valType == "diff_mse": # difference between train and test MSE
                val = parse_mse(d[-3]) - parse_mse(d[-6])
            all_fileValues.append(val)
            all_files.append(f)

            if key not in params_to_val:
                params_to_val[key] = []

            params_to_val[key].append(val)
            bar.update(i)
                
    # flip and sort values
    all_val = []
    if valType == "numGen":
        name = "numGen"
    elif valType == "test_mse":
        name = "test MSE"
    elif valType == "train_mse":
        name = "train MSE"
    elif valType == "diff_mse":
        name = "test minus train MSE"
    else:
        name = "tree size"
    for params in params_to_val:
        # get means and standard deviations
        mean_val = mean(params_to_val[params])
        std_val = stdev(params_to_val[params])
        # reassign
        params_to_val[params] = (mean_val, std_val)
        # sort means
        all_val.append(mean_val)
        val_to_params[mean_val] = params
                 
    all_val.sort()

    # return top values
    results = []    
    for i in range(len(all_val)):
        params = val_to_params[all_val[i]]
        (mean_val, std_val) = params_to_val[params]
        if i < printNum:
            print("Result " + str(i) + ": " +  str(params), "with mean", 
                  name, mean_val, "-", "std", name, std_val)
        results.append(params)
            
    if allStats:
        return results, params_to_val, val_to_params, all_val, all_files, all_fileValues
    else:
        return results, params_to_val, val_to_params, all_val
    
def parse_mse(line):
    mse = float(line.strip()[4:])
    return mse

def parse_numGen(line):
    return int(line.split("_")[0])

def parse_treeSize(line):
    return int(re.findall(r'\d+', line)[0])

def get_key(f):
    k = f[:-6]
    if k[-1] == "_":
        return f[:-7]
    else:
        return k
    
def file_read_from_tail(fname,lines):
        fname = fname
        bufsize = 8192
        fsize = os.stat(fname).st_size
        iter = 0
        with open(fname) as f:
                if bufsize > fsize:
                        bufsize = fsize-1
                        data = []
                        while True:
                                iter +=1
                                f.seek(fsize-bufsize*iter)
                                data.extend(f.readlines())
                                if len(data) >= lines or f.tell() == 0:
                                        return data[-lines:]

res = file_read_from_tail("C:/Users/nele2/Desktop/SimpleGP/experiments_numInd_noInitial_genIter\maxtime20_pop512_mr0.001_tour8_maxHeight2_cr1__topK0.1_unK-1_gen1_lr0.001_it1_oIt1_3.txt", 11)
o = parse_treeSize(res[2])
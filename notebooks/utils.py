from statistics import mean, stdev
import os
import progressbar 
import re
import numpy as np

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

def get_params_final_function(experiment_dir):
    '''
    Returns for each file: number of variables, number of +-functions, etc.
    '''
    files = os.listdir(experiment_dir)
    
    stats = []
    all_files = [] # all file names
    
    # aggregate value
    with progressbar.ProgressBar(max_value=len(files)) as bar:
        for i, f in enumerate(files):
            
            d = file_read_from_tail(experiment_dir + f, 11)
            if d == None:
                continue
            
            currStats = []
           
            func = d[-8] # final function
            tree_size = parse_treeSize(d[2])
            num_variables = parse_function_count(func, "x")
            
            # compute statistics
            currStats.append(parse_function_varDiff(func))
            currStats.append(parse_function_varDiff(func) / tree_size)
            if num_variables > 0:
                currStats.append(parse_function_varDiff(func) / num_variables)
            else:
                currStats.append(0)
            currStats.append(parse_function_count(func, "x"))
            currStats.append(parse_function_count(func, "x") / tree_size)
            currStats.append(parse_function_count(func, "+"))
            currStats.append(parse_function_count(func, "+") / tree_size)
            currStats.append(parse_function_count(func, "*"))
            currStats.append(parse_function_count(func, "*") / tree_size)
            currStats.append(parse_function_count(func, "/"))
            currStats.append(parse_function_count(func, "/") / tree_size)
            currStats.append(parse_function_constants(func, "num"))
            currStats.append(parse_function_constants(func, "num") / tree_size)
            currStats.append(parse_function_constants(func, "min"))
            currStats.append(parse_function_constants(func, "max"))
            currStats.append(parse_function_constants(func, "mean"))
            currStats.append(parse_function_constants(func, "std"))
            currStats.append(parse_function_weights(func, "num"))
            currStats.append(parse_function_weights(func, "min"))
            currStats.append(parse_function_weights(func, "max"))
            currStats.append(parse_function_weights(func, "mean"))
            currStats.append(parse_function_weights(func, "std"))
            
            stats.append(currStats)
            all_files.append(f)
            bar.update(i)
    
    return all_files, stats
            
                
def parse_function_count(line, character = "x"):
    return line.count(character) 

def parse_function_varDiff(line):
    '''
    Returns number of different variables in function.
    '''
    indices = [m.start() + 1 for m in re.finditer('x', line)]
    found = np.zeros(10)
    for i in indices:
        found[int(line[i])] = 1
    return np.count_nonzero(found == 1)

def parse_function_constants(line, statistic = "mean"):
    '''
    Computes statistics of the constants in the final function.
    
    statistic: "mean", "std", "min", "max", "num"
    '''
    brackets = 0
    prev = ""
    read = False
    currConstant = ""
    constants = []
    
    for i in line: # for each character
        if i == "[":
            brackets += 1
        elif i == "]":
            brackets -= 1
        elif brackets == 1:
            if read == True and i == ",":
                read = False
                constants.append(float(currConstant))
                currConstant = ""
            elif prev == " " and i.isdigit() and read == False:
                currConstant += i
                read = True
            elif read == True:
                currConstant += i
        prev = i
    
    if len(constants) > 0:
        # Compute statistic
        if statistic == "mean":
            return np.mean(constants)
        if statistic == "std":
            return np.std(constants)
        if statistic == "min":
            return np.min(constants)
        if statistic == "max":
            return np.max(constants)
        if statistic == "num":
            return len(constants)
    return 0
    
def parse_function_weights(line, statistic = "mean"):
    '''
    Computes statistics of the weights in the final function.
    
    statistic: "mean", "std", "min", "max", "num"
    '''
    brackets = 0
    prev = ""
    read = False
    currWeight = ""
    weights = []
    
    for i in line: # for each character
        if i == "[":
            brackets += 1
        elif i == "]":
            brackets -= 1
            if read == True:
                weights.append(float(currWeight))
                currWeight = ""
            read = False
        elif brackets == 2:
            if read == True and i == " ":
                read = False
                weights.append(float(currWeight))
                currWeight = ""
            elif (prev == " " or prev == "[") and (i.isdigit() or i == "-") and read == False:
                currWeight += i
                read = True
            elif read == True:
                currWeight += i
        prev = i
        
    if len(weights) > 0:
        # compute desired statistic
        if statistic == "mean":
            return np.mean(weights)
        if statistic == "std":
            return np.std(weights)
        if statistic == "min":
            return np.min(weights)
        if statistic == "max":
            return np.max(weights)
        if statistic == "num":
            return len(weights)
    return 0

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
                                    
f = "[-[ 1.7367447   2.00502868 -1.67595966 -0.88210146], *[-1.46825686  0.45251727 \
-0.88661354 -1.10247237], /[-0.94202238 -1.0463323  -1.42270406  0.36304734], 1.167, \
    x3, /[-0.94202238 -1.0463323  -1.42270406  0.36304734], x7, x8, *[-1.49188388 \
    -1.41455442 -0.89825119 -1.59801406], *[-1.46825686  0.45251727 -0.88661354 \
    -1.10247237], *[-1.46825686  0.45251727 -0.88661354 -1.10247237], 3.502, x7, 3.643, x7]"
from statistics import mean, stdev
import os
import progressbar 

def aggregateParams(experiment_dir):
    files = os.listdir(experiment_dir)
    
    mse_to_params = {}
    params_to_mse = {}
    
    # aggregate mse
    with progressbar.ProgressBar(max_value=len(files)) as bar:
        for i, f in enumerate(files):
            key = get_key(f)
            d = file_read_from_tail(experiment_dir + f,3)
            if d == None:
                continue
            mse = parse_mse(d[0])

            if key not in params_to_mse:
                params_to_mse[key] = []

            params_to_mse[key].append(mse)
            bar.update(i)
                
    # flip and sort mses
    all_mse = []
    for params in params_to_mse:
        # get means and standard deviations
        mean_mse = mean(params_to_mse[params])
        std_mse = stdev(params_to_mse[params])
        # reassign
        params_to_mse[params] = (mean_mse, std_mse)
        # sort means
        all_mse.append(mean_mse)
        mse_to_params[mean_mse] = params
                 
    all_mse.sort()

    # return top 5 mses
    results = []    
    for i in range(20):
        params = mse_to_params[all_mse[i]]
        (mean_mse, std_mse) = params_to_mse[params]
        print("Result " + str(i) + ": " +  str(params) + " with mean mse " + str(mean_mse) + ", std mse " + str(std_mse))
        results.append(params)
            
    return results, params_to_mse, mse_to_params, all_mse

def parse_mse(line):
    mse = float(line.strip()[4:])
    return mse

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

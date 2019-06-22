import matplotlib.pyplot as plt
import numpy as np

def errorbar_plot(values):
    x = [] 
    y =[] 
    e = []
    
    for v in values:
        x.append(v[0])
        y.append(v[1][0])
        e.append(v[1][1])
    
    x = np.array(x)
    y = np.array(y)
    e = np.array(e)
    plt.errorbar(x, y, e, linestyle='-', marker='^')
    plt.show()

def calc_means_and_stds(template_str, options, p_to_mse, two = False):
    values = []
    for o in options:
        if not two:
            key = template_str.format(o)
        else:
            key = template_str.format(o, o) # for number of backprop iterations
        values.append((o, p_to_mse[key]))
    return values
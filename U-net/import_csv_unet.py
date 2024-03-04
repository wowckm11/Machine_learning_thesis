import csv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.interpolate import make_interp_spline
import os 


def is_outlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False 
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh
def create_graphs_from_csv(filepath):
    try:
        file = filepath + "/scores.csv"
        f = open(file, 'r')
    except FileNotFoundError:
        print(f"file {file} not found")
        return
    base = []
    y_classifier = []
    y_box_reg = []
    counter = 0
    for line in csv.reader(f):
        y_classifier.append((float(line[0])))
        y_box_reg.append((float(line[1])))
        counter +=1 
        base.append(counter)

    loss_list = [y_classifier, y_box_reg]
    loss_dict = {0:'accuracy%',1:'dice score'}
    loss_name_dict = {0:'accuracy',1:'dice'}
    counter = 0
    for item in loss_list:
        plt.show()
        y = np.array(item)
        x = np.array(base)
        y = y[~is_outlier(y)]
        x = x[:len(y)]
        xnew = np.linspace(x.min(), x.max()) 

        #define spline
        spl = make_interp_spline(x, y, k=2)
        y_smooth = spl(xnew)

        #create smooth line chart 
        plt.plot(xnew, y_smooth, color='r')
        plt.scatter(x,y)
        output_path = f"{filepath}/{loss_name_dict[counter]}-line"
        title = f"{loss_dict[counter]}"
        plt.title(title)
        
        plt.xlabel('epochs')
        plt.ylabel(f"{loss_name_dict[counter]}")
        plt.savefig(output_path)
        plt.close()
        plt.show(block=False)
        plt.scatter(x,y)
        output_path = f"{filepath}/{loss_name_dict[counter]}"
        title = f"{loss_dict[counter]}"
        plt.title(title)
        
        plt.xlabel('epochs')
        plt.ylabel(f"{loss_name_dict[counter]}")
        plt.savefig(output_path)
        plt.close()
        plt.show(block=False)
        plt.hist2d(x,y,(10,10))
        output_path = f"{filepath}/{loss_name_dict[counter]}-histo"
        title = f"{loss_dict[counter]}"
        plt.title(title)
        
        plt.xlabel('epochs')
        plt.ylabel(f"{loss_name_dict[counter]}")
        plt.savefig(output_path)
        plt.close()
        plt.show(block=False)
        counter+=1

def get_all_paths():
    base_path = 'saved_models'
    list_of_full_paths = []
    path_tree = os.listdir(base_path)
    for item in path_tree:
        path = f"{base_path}/{item}"
        list_of_full_paths.append(path)
    return(list_of_full_paths)


for path in get_all_paths():
    create_graphs_from_csv(path)
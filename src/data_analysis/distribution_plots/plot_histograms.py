import os
import matplotlib.pyplot as plt
import numpy as np

# column position including the filename
columns_data_normalized_onehot = {
    'left': 1,
    'right': 2,
    'front': 3,
    'rear': 4,
    'area': 5,
    'major': 6,
    'minor': 7,
    'circ': 8,
    'ar': 11,
    'label': 53
}

csvData = np.loadtxt(
                    'data/labelled/data_normalized_onehot.csv',
                     delimiter=',',
                     skiprows=1,
                     # exclude filename column  
                     usecols=range(1, 54)
)

def plot_feature(feature_name='area', wing_region='front', bins=25):
    plt.hist([
        x[columns_data_normalized_onehot[feature_name]-1] for x in csvData \
        if x[columns_data_normalized_onehot['label']-1] == 0 and \
        x[columns_data_normalized_onehot[wing_region]-1] == 1], \
    bins=bins, label='arm', alpha=0.3)
    plt.hist([
        x[columns_data_normalized_onehot[feature_name]-1] for x in csvData \
        if x[columns_data_normalized_onehot['label']-1] == 1 and \
        x[columns_data_normalized_onehot[wing_region]-1] == 1], \
        bins=bins, label='zea', color='green', alpha=0.3)

    plt.title(f'Distribution of wing {feature_name}, {wing_region} wings')
    plt.legend()
    plt.savefig(f'{feature_name}_{wing_region}wings.png')

plot_feature('circ', 'front')
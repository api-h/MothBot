import sys
import os
import matplotlib.pyplot as plt
import numpy as np

# plot example data
csvData = np.loadtxt('data/labelled/data_normalized_onehot.csv',
                     delimiter=',',
                     skiprows=1,
                     usecols=range(1, 54))

labels = np.loadtxt('data/labelled/data_normalized_onehot.csv',
                    delimiter=',',
                    dtype=np.str0)

labels = labels[0, 1:]

print(labels)

# print(csvData.shape)

# filter data by label
filtered = csvData[csvData[:, 0] == 1]
filtered = filtered[filtered[:, 2] == 1]

# print(filtered.shape)

for j in range(4, 53):
    data = [(filtered[i, j], filtered[i, -1])
            for i in range(filtered.shape[0])]
    data.sort(key=lambda x: x[0])

    fig = plt.scatter(range(filtered.shape[0]), [x[0] for x in data],
                      c=[x[1] for x in data])
    
    plt.title(labels[j])

    # plt.show()
    # create file

    if not os.path.exists('plots'):
        os.makedirs('plots')

    plt.savefig('plots/' + labels[j].lower() + '.png')
    plt.clf()
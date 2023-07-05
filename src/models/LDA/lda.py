import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

csvData = np.loadtxt('data/labelled/data_normalized_onehot.csv',
                     delimiter=',',
                     skiprows=1,
                     usecols=range(1, 54))

x, y = np.split(csvData, [-1], 1)  # pylint: disable=unbalanced-tuple-unpacking
y = y.ravel()

lda = LinearDiscriminantAnalysis(n_components=1)
lda_transformed = list(map(lambda x: x[0], lda.fit_transform(x, y)))

def plot():
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.scatter(lda_transformed, [0] * len(lda_transformed), c=y)

    plt.show()

# plot()
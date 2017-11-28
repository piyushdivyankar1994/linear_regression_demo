import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from sklearn import datasets, linear_model

# #############################################################################
# Plot the figure
def plot_figs(fig_num, elev, azim, X_train, clf):
    fig = plt.figure(fig_num, figsize=(4, 3))
    plt.clf()
    ax = Axes3D(fig, elev=elev, azim=azim)

    ax.scatter(X_train[:, 0], X_train[:, 1], y_train, c='k', marker='+')
    ax.plot_surface(np.array([[-.1, -.1], [.15, .15]]),
                    np.array([[-.1, .15], [-.1, .15]]),
                    clf.predict(np.array([[-.1, -.1, .15, .15],
                                          [-.1, .15, -.1, .15]]).T
                                ).reshape((2, 2)),
                    alpha=.5)
    ax.set_xlabel('X_1')
    ax.set_ylabel('X_2')
    ax.set_zlabel('Y')
    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])


diabaties = datasets.load_diabetes()

# Print description of the diabetes dataset.
print diabaties['DESCR']

indices = (0,2)

print "\n==============\nWe will be using two variables here {} and {}\n".format(diabaties.feature_names[indices[0]], diabaties.feature_names[indices[1]])
X_train = diabaties.data[:-20, indices]
y_train = diabaties.target[:-20]
x_test = diabaties.data[-20: , indices]
y_test = diabaties.target[-20:]

lin_model = linear_model.LinearRegression()
lin_model.fit(X_train, y_train)

# Plotting this fitted curve
plot_figs(1, 43.5, -110, X_train, lin_model)
plot_figs(2, -.5, 0, X_train, lin_model)
plot_figs(3, -.5, 90, X_train, lin_model)

plt.show()

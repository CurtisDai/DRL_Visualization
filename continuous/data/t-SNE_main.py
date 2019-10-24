import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold
import pandas as pd
FILEPATH = ["./data.txt","./data1.txt","./data2.txt"]
X = []
Y = []

for path in FILEPATH:
    file = open(path)
    line = file.readline()
    while line:
        data = line.split("\t")
        x = np.array([float(i) for i in data[0].strip().split()])
        X.append(x)
        Y.append(data[1].strip("\n"))
        line = file.readline()
    file.close()



def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(label[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig


tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
X_tsne = tsne.fit_transform(X)

print("Org data dimension is {}.Embedded data dimension is {}".format(len(X), X_tsne.shape[-1]))

'''嵌入空间可视化'''
# fig = plot_embedding(X_tsne, Y,'t-SNE embedding of the digits (time %.2fs)')
# plt.show(fig)
x_min, x_max = np.min(X_tsne, 0), np.max(X_tsne, 0)
data = (X_tsne - x_min) / (x_max - x_min)

plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

#不同类别用不同颜色和样式绘图
for i in range(len(Y)):
    if Y[i] == "LEFT":
        plt.plot(X_tsne[i][0], X_tsne[i][1], 'r.')
    elif Y[i] == "RIGHT":
        plt.plot(X_tsne[i][0], X_tsne[i][1], 'g.')
    else:
        plt.plot(X_tsne[i][0], X_tsne[i][1], 'b.')


plt.show()

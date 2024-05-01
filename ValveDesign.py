import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from Optimization.Algorithm import classy as cl, sampling as sp, geneticalg as ga

"""""""""
This file, along with ValveArchive contain functions used in the process of optimizing the efficiency of an engine
when forced to use blackbox methods.

Included is methods for fitting models, optimizing a model, and plotting models.

The current code is set to plot an experiment of interpolating between quadratic models. This was not a successful method.
"""""""""

def quadfit(input, para, p):
    a1, a2, a3, a4, a5, a6, a7, a8, a9, a10 = input
    x0, y0, z0, null = para.parameter
    A = np.array([[a1, a2, a3],
                  [0, a4, a5],
                  [0, 0, a6]])
    b = np.array([a7, a8, a9])
    c = a10
    dist = []
    for k in range(len(para.data)):
        x = np.array([para.data[k][0] - x0, para.data[k][1] - y0, para.data[k][2] - z0])
        dist.append((x.dot(np.dot(A, x)) + np.dot(x, b) + c - para.data[k][3]) ** 2)
    return np.array(dist).sum()


def quadint(input, data, parameter, center):
    x = []
    for j in range(len(input)):
        a1, a2, a3 = parameter[j]
        x.append(a1 * (input[j] - center[j]) ** 2 - a2 * (input[j] - center[j]) - a3)
    x = np.array(x)
    a = []
    for j in range(int(len(data) / 10)):
        a1, a2, a3, a4, a5, a6, a7, a8, a9, a10 = data[10 * j:10 * j + 10]
        A = np.array([[a1, a2, a3],
                      [0, a4, a5],
                      [0, 0, a6]])
        b = np.array([a7, a8, a9])
        c = a10
        a.append(np.dot(x, np.dot(A, x)) + np.dot(x, b) + c)
    a1, a2, a3, a4, a5, a6, a7, a8, a9, a10 = a
    A = np.array([[a1, a2, a3],
                  [0, a4, a5],
                  [0, 0, a6]])
    b = np.array([a7, a8, a9])
    c = a10
    return np.dot(x, np.dot(A, x)) + np.dot(x, b) + c


data = [[3.00, 1.90, 3.10, .249],
        [4.20, 1.40, 3.30, .034],
        [4.20, 2.10, 2.90, .11],
        [4.66, 2.20, 3.78, .039],
        [2.80, 1.95, 3.17, .234],
        [3.20, 1.70, 2.90, .203],
        [3.60, 2.40, 3.70, .226],
        [3.90, 2.00, 2.70, .171],
        [3.129, 2.256, 3.39, .231],
        [2.22, 2.71, 3.00, .092],
        [3.50, 2.50, 3.40, .486],
        [2.90, 1.892, 4.00, 0.075],
        [3.70, 2.70, 3.40, 0.371],
        [4.48, 3.00, 3.38, 0.065],
        [2.20, 1.60, 1.20, .049],
        [2.60, 2.80, 1.80, .027],
        [0.74, 1.91, 2.73, .060],
        [1.95, 2.37, 3.49, .191],
        [4.80, 1.00, 2.60, .009],
        [3.50, 1.80, 1.90, .132],
        [4.80, 2.80, 1.10, .180],
        [2.20, 2.70, 4.00, .082],
        [3.60, 2.40, 3.642, 0.252],
        [3.558, 2.577, 3.427, 0.512],
        [3.46, 2.80, 2.76, 0.070],
        [3.68, 2.60, 3.53, 0.608],
        [3.42, 2.61, 3.44, 0.353],
        [3.62, 2.654, 3.434, 0.440]]

coefficients = [2.06908405, 0.46854785, -1.56235294, -12.60713025,
                -0.61996572, -1.25174086, 1.30368313, -2.66997828,
                -1.18665068, -0.11304734, 8.32347227, -2.40108638, 8.3428952, -18.82908977,
                4.33384832, 6.76789994, 2.07001031, -3.44641262,
                4.88948332, 0.51740523, -11.65143996, 4.15960153, -15.12750265, 15.06995419,
                -15.81171431, -17.10198024, -0.91725695, 1.54165064,
                -10.81728058, 5.091137, -25.23736794, 2.63846225, -4.33721538, 90.18813688,
                3.40919536, -6.80759074, -8.84148977, 20.10136195,
                -2.74280783, -2.74442583, 3.25083397, 1.92780122, -1.18162757, -23.67038689,
                1.87724267, -1.07295839, 2.16807974, -1.3704851,
                -0.57555078, 6.23845546, 7.87882978, -2.67228624, 5.48181187, -23.11424799,
                3.48391624, 2.99715656, 2.4900835, -3.17746436,
                2.55488518, 0.88164636, 0.63353277, -0.90218675, 4.92012712, 0.97650419, 3.66919693,
                1.2192744, 0.06608873, 0.41602471, 4.84756024, 0.8863286, -3.54827721, -0.47812439, 2.45831419,
                10.2754864, -2.05395599,
                3.72177897, -0.01076352, -0.89850131, 6.09700086, 0.49379961, -2.11933361, 2.64338843, 4.8181471,
                10.39634914, 8.3058236,
                3.35882241, 4.37712437, 8.64461836, 2.80904419, 0.80883657, -1.00411219, 0.16482177, 4.83405394,
                1.23282851, 5.6533311,
                0.21280897, 0.63739515, 1.04285128, 0.83301942, 0.56372751]

parameter = [[0.0063, .9832, -0.0209], [-0.003, .9853, .017], [-0.013, 0.994, 0.0046]]

x = []
y = []
z = []
f = []
for k in range(len(data)):
    x.append(data[k][0])
    y.append(data[k][1])
    z.append(data[k][2])
    f.append(data[k][3])

f = np.array(f)

mesh = 15
inputx = np.linspace(2, 5, mesh)
inputy = np.linspace(1, 3, mesh)
inputz = np.linspace(1, 4, mesh)

input = []
for k in range(mesh):
    for j in range(mesh):
        for i in range(mesh):
            input.append([inputx[k], inputy[j], inputz[i]])

input = np.array(input)
output = []
inputx = []
inputy = []
inputz = []
for k in range(mesh ** 3):
    if abs(input[k][0] - input[k][2]) <= 1.1 and np.linalg.norm(
            [input[k][0] - 3.68, input[k][1] - 2.60, input[k][2] - 3.53]) < 5:
        inputx.append(input[k][0])
        inputy.append(input[k][1])
        inputz.append(input[k][2])
        output.append(quadint([input[k][0], input[k][1], input[k][2]], coefficients, parameter, [3.68, 2.60, 3.53]))
output = np.array(output)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.view_init(45, 60)

ax.scatter(inputx, inputy, inputz, facecolors=cm.Reds(output.flatten()))
ax.scatter(x, y, z, facecolors=cm.Greens(f * 2))
ax.scatter(3.7324388, 2.66947628, 3.61971185, facecolors=cm.Blues(1.20))

plt.show()

s = 0
ndata = []
for j in range(len(data)):
    if np.linalg.norm(np.array([data[j][0] - 3.68, data[j][1] - 2.60, data[j][2] - 3.53])) < 1:
        ndata.append(data[j])

for k in range(len(ndata)):
    s += (quadint([ndata[k][0], ndata[k][1], ndata[k][2]], coefficients, parameter, [3.68, 2.60, 3.53]) - ndata[k][3]) ** 2
print(s)

"""""""""
def true(input):
    return True


input = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
ndata = []
for j in range(len(data)):
    if np.linalg.norm(np.array([data[j][0] - 3.68, data[j][1] - 2.60, data[j][2] - 3.53])) < 1:
        ndata.append(data[j])

a = []
for j in range(len(ndata)):
    para = cl.para(0.0001, 0.19, ndata, ndata[j], true)
    pr = cl.funct(quadfit, 'min', '', input, para, 1000)
    a.append([ndata[j]])
    a[j].append(ga.genetic(pr))

print(a)

fit = []
for j in range(10):
    fdata = []
    for i in range(len(ndata)):
        fdata.append([a[i][0][0], a[i][0][1], a[i][0][2], a[i][1][j]])
    para = cl.para(0.0001, 0.19, fdata, [3.68, 2.60, 3.53, 0.608], true)
    pr = cl.funct(quadfit, 'min', '', input, para, 1000)
    fit.append(ga.genetic(pr))

print(fit)
"""""""""
"""""""""
def boundary(input):
    if 2 <= input[0] <= 5 and 1 <= input[1] <= 3 and 1 <= input[2] <= 4 and abs(input[0] - input[2]) <= 1.1 and np.linalg.norm(np.array(input) - np.array([3.68, 2.6, 3.53])) < 1 / 8:
        return True
    else:
        return False


input = [3.68, 2.60, 3.53]
para = cl.para(0.0001, 0.19, coefficients, [3.68, 2.60, 3.53], boundary)
pr = cl.funct(quadint, 'max', '', input, para, 1000)

ga.genetic(pr)


# [3.7324388  2.66947628 3.61971185]___1.1962277265469559
"""""""""

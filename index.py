# Author by: Victor Yip github/victorswkyip
# Date Start: Aug. 19, 2019
# Date Submitted: Aug. 26, 2019
# Submitted To: thirdhandai[at]gmail[dot]com
# Brief: Code Submission for Python Developer Test at SkateScribe

# ----------------------------------------------------------------------------------------------------------------------
# Import Libraries
# ----------------------------------------------------------------------------------------------------------------------

import cv2
import pandas as pd
import plotly.graph_objects as go
import numpy as np


# ----------------------------------------------------------------------------------------------------------------------
# Define Functions
# ----------------------------------------------------------------------------------------------------------------------

def preSmooth(data):
    df = pd.DataFrame(data.values)
    blur = cv2.GaussianBlur(df.values, (5, 5), 0)  # Gaussian Blur with a 5x5 Kernel and standard deviation of 0
    img = pd.DataFrame(blur)
    return img


def edgeDetect(data, axis):
    # (1) Calculate gradient of the amplitude of the data
    # (2) Evaluate gradient based on a threshold value that is sensitive enough to return 5 segments
    # (3) Eliminated double counted positives
    # (4) Return the index location of the detections

    if axis == 'x':
        switch = 1
    elif axis == 'y':
        switch = 0
    else:
        print('error: no axes specified')

    gradient = data.pct_change(axis=switch)  # (1)
    minimum_gradient_threshold = 1000  # arbitrary tweak
    nearest_neighbour_threshold = 10  # arbitrary tweak

    result = np.where(gradient >= minimum_gradient_threshold)  # get indices (2)
    final_result = sorted(result[switch])  # data massage

    # Assume that, if detected edges are nearby each other within a certain threshold, then they are same edge
    section_index = []
    for x in range(len(final_result)):  # (3)
        if x == 0:
            section_index.append(final_result[x])
        else:
            if final_result[x] - final_result[x - 1] > nearest_neighbour_threshold:
                section_index.append(final_result[x])
    return section_index  # (4)


def postSmooth(data):
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])  # arbitrary kernel that emphasizes the centre pixel
    img = pd.DataFrame(cv2.blur(cv2.filter2D(data.values, -1, kernel), (5, 5)))
    return img


def plot3D(data):
    fig = go.Figure(data=[go.Surface(z=data.values)])
    fig.update_layout(title='3D Scan Reconstruction', autosize=True,
                      width=600, height=600,
                      margin=dict(l=65, r=50, b=65, t=90))
    fig.show()


def dataSegment(data, index):
    section = [0, 0, 0, 0, 0]  # a non-ideal method to initialize a store array, instead reference len(index)
    for x in range(len(section)):
        if x == 0:
            section[x] = data.loc[:, x:index[x] - 1]
        elif x == 4:  # ideally reference last element in the array
            section[x] = data.loc[:, index[x - 1]:, ]
        else:
            section[x] = data.loc[:, index[x - 1]:index[x] - 1]
    return section


def plot2D(data):
    # 2D Grayscale Heatmap
    layout = go.Layout(
        xaxis=go.layout.XAxis(
            tickmode='linear',
            tick0=0,
            dtick=50,
        ),
        yaxis=dict(
            scaleanchor="x",
            scaleratio=.1,
        )
    )
    colorscale = [
        [0, "rgb(0, 0, 0)"],
        [0.1, "rgb(0, 0, 0)"],

        [0.1, "rgb(20, 20, 20)"],
        [0.2, "rgb(20, 20, 20)"],

        [0.2, "rgb(40, 40, 40)"],
        [0.3, "rgb(40, 40, 40)"],

        [0.3, "rgb(60, 60, 60)"],
        [0.4, "rgb(60, 60, 60)"],

        [0.4, "rgb(80, 80, 80)"],
        [0.5, "rgb(80, 80, 80)"],

        [0.5, "rgb(100, 100, 100)"],
        [0.6, "rgb(100, 100, 100)"],

        [0.6, "rgb(120, 120, 120)"],
        [0.7, "rgb(120, 120, 120)"],

        [0.7, "rgb(140, 140, 140)"],
        [0.8, "rgb(140, 140, 140)"],

        [0.8, "rgb(160, 160, 160)"],
        [0.9, "rgb(160, 160, 160)"],

        [0.9, "rgb(180, 180, 180)"],
        [1.0, "rgb(180, 180, 180)"]
    ]

    fig = go.Figure(
        layout=layout
    )
    fig.add_trace(go.Heatmap(
        z=data,
        colorscale=colorscale,
    ))
    fig.show()


def getRadius(data, scan_size):

    # For simplicity, we'll assume that the dimples are spherical, even though they appear elliptical in 3D plots
    # Assume that that the dimples are cross-sectionally uniform along x, calculating one slice is as good as another

    gradient = data.pct_change()
    laplace = gradient.pct_change()
    localized_laplace = laplace.loc[944-scan_size:944+scan_size,:]  # 944 is from an initial calc for centre point, ==>
    # ideal automation would iterate to find centre point, but I'll guide it since I have already seen the 2D heatmaps

    # find indices where there are asymptotes, aka dimple centre point
    result = np.where(pd.DataFrame.isnull(localized_laplace))
    final_result = sorted(result[0])  # data massage

    # Geometry calculations
    dimple_yedge_start = min(final_result)
    dimple_yedge_end = max(final_result)
    dimple_diameter = dimple_yedge_end - dimple_yedge_start
    dimple_radius = dimple_diameter / 2

    return dimple_radius


def getRadiusUncertainty():

    # Let's just evaluate the integrity of our answer via Radius calculation by sweeping over a scan size
    N = 100
    t = np.linspace(0, 4999,  N)
    y = np.linspace(0, 0, N)
    for x in range(len(t)):
        y[x] =getRadius(z3_data,t[x])

    fig = go.Figure(data=go.Scatter(x=t, y=y, mode='lines+markers'))

    fig.update_layout(
    title = go.layout.Title(
        text="Radius End Behaviour wrt Scan Size",
    ),
    xaxis = go.layout.XAxis(
        title=go.layout.xaxis.Title(
            text="Scan Size or Delta Slices [pixels]",
        )
    ),
    yaxis = go.layout.YAxis(
        title=go.layout.yaxis.Title(
            text="Calculated 'Radius' [pixels]",
            )
        )
    )
    fig.show()


# ----------------------------------------------------------------------------------------------------------------------
# Program Logic Starts Here -- Main Tasks
# ----------------------------------------------------------------------------------------------------------------------

z0_data = pd.read_csv("point-cloud.csv")  # import given dataset
# there are 800 columns, let's call this the x-axis, and ~5000 rows let's call that the y-axis
image = preSmooth(z0_data)
plot3D(image)  # question -a

edges = edgeDetect(image, 'x')  # partial derivative wrt x
print('The program has detected edges at indices: ' + str(edges))  # question -b

z1_data = dataSegment(image, edges)  # Subset the data at the edges
z2_data = []

print('Hope you enjoy these plots. Incoming..... (This will take ~3 minutes. We appreciate your patience.)')
for x in range(len(z1_data)):  # Generate 3D plots of each subset, compare side to side
    plot3D(z1_data[x])  # 3D plot
    plot2D(z1_data[x])  # question -c
    z2_data.append(postSmooth(z1_data[x]))  # apply post-segmentation smoothing  question -d
    plot2D(z2_data[x])

z3_data = z1_data[1]  # based on visual inspection of the 2D heatmaps, z1_data[1] has the best image quality for dimples
z4_data = z2_data[1]  # smoothened z3_data
print('The radius of the dimple(s) was found to be somewhere between ' + str(getRadius(z4_data, 500)) +
      ' and '+ str(getRadius(z3_data, 500)) + ' pixels')  # bonus question -e

answer = (getRadius(z3_data, 500)+getRadius(z4_data, 500))/2  # averaged
error = (getRadius(z3_data, 500)-getRadius(z4_data, 500))/2

print('in other words: radius = ' + str(answer) + ' +/- ' + str(error) + ' pixels')
print('Please wait.... last plot')
getRadiusUncertainty()  # sanity check

# ----------------------------------------------------------------------------------------------------------------------
# End of Code
# ----------------------------------------------------------------------------------------------------------------------

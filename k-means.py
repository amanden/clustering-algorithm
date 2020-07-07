"""
    Filename : K-means.py
    Author : Aswathi Kunnumbrath Manden

    The implementation of k-means clustering algorithm.
    This file prints the cendroids and SSE(Sum of squared errors)
    for different values of k (from k = 2 to k = 20) and plots the graph
    between k and SSE to find the optimal or knee value of k.

    Takes the path of the excel file containing the data as
    the command line argument.

"""
import pandas as pd
import random
import math
import matplotlib.pyplot as plt
import sys


def main():
    """
     The main method of execution.
     
     Plots the graph between k and SSE after finding the clusters. 
    """
    file = sys.argv[1]
    dataset = pd.read_excel(file, sheet_name="Clustering", index_col=0);

    x_axis = []
    y_axis = []
    # for k from 2 to 20
    for k in range(2, 22, 2):
        print("For k = ", k)
        centroids = []
        for i in range(k):
            centroid = getRandomCentroids(dataset, k)
            centroids.append(centroid)
        oldCentroids = None
        dataset["centroid"] = 0

        count = 1
        while not shouldStop(oldCentroids, centroids, count):
            oldCentroids = centroids
            assignPoint(dataset, centroids)
            centroids = getCentroids(dataset, k)
            count += 1

        print(" Centroid found is: ", centroids)
        sse = getSSE(k, dataset, centroids)
        print(" Sum of squared errors: ", sse)

        x_axis.append(k)
        y_axis.append(sse)

    # Plots the graph between k and SSE
    plt.plot(x_axis, y_axis)
    plt.title('K means- SSE vs k', fontsize=16)
    plt.xlabel('k', fontsize=14)
    plt.ylabel('sum of squared errors', fontsize=14)
    plt.show()


def getSSE(k, dataset, centroids):
    """
     Returns the sum of squared errors.
    """
    sse = 0
    for i in range(k):
        df = dataset.loc[dataset['centroid'] == i]
        data = df.loc[:, df.columns != 'centroid'].values.tolist()
        for items in data:
            sse += (distance(items, centroids[i]) ** 2)
    return sse


def shouldStop(oldCentroids, centroids, count):
    """
     Returns true when there is no change in the centroids calculated in the previous iteration or 
     when the iterations count exceed a limit( here 100).
    """
    if count > 100:
        return True
    return oldCentroids == centroids


def getCentroids(dataset, k):
    """
     Returns the list of centroids.
     Centroids are calculated as the mean of all the points assigned to it.

    """
    centroids = []
    col = []

    # for data of dimension = 10
    for n in range(10):
        col.append(dataset.columns[n])
    for i in range(k):
        df = dataset.loc[dataset['centroid'] == i]
        if df.size != 0:
            mean = df[col].mean()
            centroids.append(mean.to_list())
        else:
            centroid = getRandomCentroids(dataset, k)
            centroids.append(centroid)
    return centroids


def assignPoint(ds, centroids):
    """
     Based on the centroids found, assign each point to the nearest centroid
    """
    li = ds.loc[:, ds.columns != 'centroid'].values.tolist()
    id = 1
    for items in li:
        minDist = math.inf
        for i in range(len(centroids)):
            currDist = distance(items, centroids[i])
            if currDist < minDist:
                ds.at[id, "centroid"] = i
                minDist = currDist
        id += 1


def distance(item, centroid):
    """
     Returns the euclidean distance between the 2 points, item and the centroid
    """
    dist = 0
    for i in range(len(item)):
        dist = dist + (item[i] - centroid[i]) ** 2
    dist = dist ** 0.5
    return dist


def getRandomCentroids(ds, k):
    """
     Returns random centroid values.

     Centroids are initially assigned randomly from the range of
     minimum to maximum values in the dataset.
    """
    ds = ds.loc[:, ds.columns != 'centroid']
    centroid = []
    for columns in ds:
        max = ds[columns].max()
        min = ds[columns].min()

        randomCentroid = min + ((max - min) * random.random())
        centroid.append(randomCentroid)
    return centroid


if __name__ == '__main__':
    main()

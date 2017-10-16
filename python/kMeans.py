#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Credit: Machine Learning in Action: Chapter 10
#
# Aaron LI
# 2015/06/23
#

"""
k-means clustering algorithm
"""


import numpy as np


def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))
        dataMat.append(fltLine)
    return np.array(dataMat)


def distEclud(vecA, vecB):
    return np.sqrt(np.sum(np.power(vecA - vecB, 2)))


def randCent(dataSet, k):
    n = np.shape(dataSet)[1]
    centroids = np.zeros((k, n))
    for j in range(n):
        minJ = np.min(dataSet[:, j])
        rangeJ = float(np.max(dataSet[:, j]) - minJ)
        centroids[:, j] = minJ + rangeJ * np.random.rand(k)
    return centroids


def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = np.shape(dataSet)[0]
    clusterAssment = np.zeros((m, 2))
    centroids = createCent(dataSet, k)
    clusterChanged = True
    iterations = 0
    while clusterChanged:
        clusterChanged = False
        iterations += 1
        for i in range(m):
            minDist = np.inf
            minIndex = -1
            # to find the nearest centroid
            for j in range(k):
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist**2
        #print(centroids)
        for cent in range(k):
            ptsInClust = dataSet[np.nonzero(clusterAssment[:, 0] == cent)]
            centroids[cent, :] = np.mean(ptsInClust, axis=0)
    result = {
            'k': k,
            'centroids': centroids,
            'labels': clusterAssment[:, 0].astype(int),
            'distance2': clusterAssment[:, 1],
            'accessment': clusterAssment,
            'iterations': iterations
    }
    return result


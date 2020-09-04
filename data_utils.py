#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 20:26:35 2018

@author: subhayanmukherjee
"""


import numpy as np


def readShortComplex(fileName, width=1):
    return np.fromfile(fileName,'>i2').astype(np.float).view(np.complex).reshape(-1, width)

def readFloatComplex(fileName, width=1):
    return np.fromfile(fileName,'>c8').astype(np.complex).reshape(-1, width)

def readFloat(fileName, width=1):
    return np.fromfile(fileName,'>f4').astype(np.float).reshape(-1, width)

def writeShortComplex(fileName,data):
    out_file = open(fileName, 'wb')
    data.copy().view(np.float).astype('>i2').tofile(out_file)
    out_file.close()

def writeFloatComplex(fileName,data):
    out_file = open(fileName, 'wb')
    data.astype('>c8').tofile(out_file)
    out_file.close()

def writeFloat(fileName,data):
    out_file = open(fileName, 'wb')
    data.astype('>f4').tofile(out_file)
    out_file.close()

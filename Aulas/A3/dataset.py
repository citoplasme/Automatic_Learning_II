#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

class Dataset:
    
    # constructor
    def __init__(self, filename = None, X = None, Y = None):
        if filename is not None:
            self.readDataset(filename)
        elif X is not None and Y is not None:
            self.X = X
            self.Y = Y
        else:
            self.X = None
            self.Y = None
        
    def readDataset(self, filename, sep = ","):
        data = np.genfromtxt(filename, delimiter=sep)
        self.X = data[:,0:-1]
        self.Y = data[:,-1]
        
    def getXy (self):
        return self.X, self.Y
    

def test():
    d = Dataset("lr-example1.data")
    print(d.getXy())
    
test()
        

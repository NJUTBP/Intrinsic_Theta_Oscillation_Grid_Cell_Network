# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 2021

@author: Tao WANG

Description: Analyzing data for Figure 1c-f
"""

#### importing packages ####
import numpy as np
from matplotlib import pyplot as plt

def main():
    #### selecting the example ####
    TrialID=0
    LoadPath='../Data/DG_Default/sigmaEI30/sigmaIE30/TauNmda100/Fext1.8'

    SpkE=np.load(LoadPath+'/SpikeTimeE_'+str(TrialID)+'.npz')['arr_0']
    SpkI=np.load(LoadPath+'/SpikeTimeI_'+str(TrialID)+'.npz')['arr_0']

    NE=1024
    NI=256
    Nbins=64
    Tbins=140
    Nedges=np.linspace(0,360.0,Nbins+1)
    Tedges=np.linspace(0,7000,Tbins+1)

    NbinsZ=32
    TbinsZ=100
    NedgesZ=np.linspace(180,330,NbinsZ+1)
    TedgesZ=np.linspace(3000,5000,TbinsZ+1)

    histE,_,_=np.histogram2d(SpkE[:,1],SpkE[:,0],bins=[Nedges,Tedges])
    histI,_,_=np.histogram2d(SpkI[:,1],SpkI[:,0],bins=[Nedges,Tedges])
    histE=histE/(7/Tbins)/(NE/Nbins)
    histI=histI/(7/Tbins)/(NI/Nbins)

    histEZ,_,_=np.histogram2d(SpkE[:,1],SpkE[:,0],bins=[NedgesZ,TedgesZ])
    histEZ=histEZ/(2/TbinsZ)/(NE*(150/360)/NbinsZ)
    print(np.max(histE))
    print(np.max(histI))
    print(np.max(histEZ))

    fig,ax=plt.subplots(figsize=(12,6))
    ax.imshow(histEZ,interpolation='gaussian')
    plt.show()

    np.savez_compressed('../Data/DA_Figure1c-f/Hist_normal.npz',histE=histE,histI=histI,Nedges=Nedges,Tedges=Tedges)
    np.savez_compressed('../Data/DA_Figure1c-f/Hist_zoomin.npz',histEZ=histEZ,NedgesZ=NedgesZ,TedgesZ=TedgesZ)

if __name__ == '__main__':
    main()
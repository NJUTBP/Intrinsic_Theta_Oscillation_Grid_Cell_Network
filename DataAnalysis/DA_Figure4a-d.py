# -*- coding: utf-8 -*-
"""
@author: Tao WANG, Ziqun WANG
Description: Analyzing data for Figure 4a-d
"""

#### importing packages ####
import numpy as np
from matplotlib import pyplot as plt

def main():
    MultiInput=1.0
    TauGaba=10
    TNFreq=np.zeros((11,2))
    TNFreq[:,0]=np.array([50,60,70,80,90,100,110,120,130,140,150])
    j=0
    for TauNmda in [50,60,70,80,90,100,110,120,130,140,150]:
        Freq=np.loadtxt('../Data/DG_ConstBack/IE1600II1400/Freq_MI'+str(MultiInput)+'_TN'+str(TauNmda)+'_TG'+str(TauGaba)+'.txt')
        # print(Freq)
        TNFreq[j,1]=Freq
        j+=1
        
    MultiInput=1.0
    TauNmda=100
    TGFreq=np.zeros((11,2))
    TGFreq[:,0]=np.array([5,6,7,8,9,10,11,12,13,14,15])
    k=0
    for TauGaba in [5,6,7,8,9,10,11,12,13,14,15]:
        Freq=np.loadtxt('../Data/DG_ConstBack/IE1600II1400/Freq_MI'+str(MultiInput)+'_TN'+str(TauNmda)+'_TG'+str(TauGaba)+'.txt')
        # print(Freq)
        TGFreq[k,1]=Freq
        k+=1

    print(TNFreq,TGFreq)
    np.savez_compressed('../Data/DA_Figure4a-d/Freqs.npz',TNFreq=TNFreq,TGFreq=TGFreq)

if __name__ == '__main__':
    main()
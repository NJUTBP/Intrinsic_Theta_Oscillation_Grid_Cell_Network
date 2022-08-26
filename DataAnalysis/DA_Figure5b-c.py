# -*- coding: utf-8 -*-
"""
@author: Tao WANG, Ziqun WANG
Description: Analyzing data for Figure 5b-c
"""

#### importing packages ####
import numpy as np
from matplotlib import pyplot as plt

IEs=[12,18,24,30,36]
Taus=[75,100,125,150]
DCs=[0,4,8,12,16,20,24]
HWs=[0,4,8,12,16,20,24,28,32]

def main():
    StdPV=np.zeros((5,4,7000))
    MeanEnF=np.zeros((5,4))
    StdEnF=np.zeros((5,4))
    NumSpkInTheta=np.zeros((5,4))
    for i in range(5):
        ie=IEs[i]
        for j in range(4):
            tau=Taus[j]

            Path='../Data/DG_Inhibition/sigmaIE'+str(ie)+'/TauNmda'+str(tau)+'/DevtoCenter0/HalfWidth0'
            PVA=np.load(Path+'/AllPVAL.npz')['FinalPVA']
            EnF=np.load(Path+'/energyfractionintheta.npz')['EF']
            SpkF=np.load(Path+'/spikepeakfs.npz')['spikepeakf']
            MeanEnF[i,j]=np.mean(EnF)
            StdEnF[i,j]=np.std(EnF)
            NumSpkInTheta[i,j]=np.sum((4<=SpkF)*(SpkF<=12))
            StdPV[i,j,:]=np.std(PVA,axis=1)

    np.savez_compressed('../Data/DA_Figure5b-c/TauIE.npz',MeanEnF=MeanEnF,StdEnF=StdEnF,NumSpkInTheta=NumSpkInTheta,StdPV=StdPV)

if __name__ == '__main__':
    main()
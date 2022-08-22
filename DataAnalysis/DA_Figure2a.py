# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 2021

@author: Tao WANG

Description: Analyzing data for Figure 2a
"""

#### importing packages ####
import numpy as np
from matplotlib import pyplot as plt

def main():
    PVIdx=1
    TrialID=8
    LoadPath='../Data/DG_Default/sigmaEI30/sigmaIE30/TauNmda100/Fext1.8'
    PVs=np.load(LoadPath+'/PVs_'+str(TrialID)+'.npz')['arr_0']
    SpikeTimeE=np.load(LoadPath+'/SpikeTimeE_'+str(TrialID)+'.npz')['arr_0']
    
    FinalPV1=PVs[:,1]/2
    FinalPV2=(PVs[:,1]+360)/2
    for i in range(1,FinalPV1.size):
        DeltaAngle1=np.abs(FinalPV1[i]-FinalPV1[i-1])
        DeltaAngle2=np.abs(FinalPV2[i]-FinalPV1[i-1])
        DeltaAngle1=(DeltaAngle1>180) and (360-DeltaAngle1) or DeltaAngle1
        DeltaAngle2=(DeltaAngle2>180) and (360-DeltaAngle2) or DeltaAngle2
        if DeltaAngle2<DeltaAngle1:
            FinalPV1[i],FinalPV2[i]=FinalPV2[i],FinalPV1[i]

    if PVIdx==1:
        FinalPV=FinalPV1
    else:
        FinalPV=FinalPV2
    SpikeInBump=np.array([[0,FinalPV[0]]])
    for i in range(SpikeTimeE.shape[0]):
        DeltaTime=np.abs(SpikeTimeE[i,0]-PVs[:,0])
        Idx=np.where(DeltaTime==np.min(DeltaTime))[0][0]
        if FinalPV[Idx]<60:
            if (360+(FinalPV[Idx]-60))<SpikeTimeE[i,1] or SpikeTimeE[i,1]<(FinalPV[Idx]+60):
                SpikeInBump=np.append(SpikeInBump,np.array([SpikeTimeE[i,:]]),axis=0)
        elif FinalPV[Idx]<=300:
            if (FinalPV[Idx]-60)<SpikeTimeE[i,1] and SpikeTimeE[i,1]<(FinalPV[Idx]+60):
                SpikeInBump=np.append(SpikeInBump,np.array([SpikeTimeE[i,:]]),axis=0)
        else:
            if (FinalPV[Idx]-60)<SpikeTimeE[i,1] or SpikeTimeE[i,1]<(FinalPV[Idx]+60-360):
                SpikeInBump=np.append(SpikeInBump,np.array([SpikeTimeE[i,:]]),axis=0)

    np.savez_compressed('../Data/DA_Figure2a/SpikeInBump_'+str(TrialID)+'.npz',SpikeInBump=SpikeInBump,PVtime=PVs[:,0],FinalPV=FinalPV)

if __name__ == '__main__':
    main()
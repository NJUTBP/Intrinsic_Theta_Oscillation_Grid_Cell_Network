# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 2021

@author: Tao WANG

Description: Analyzing data for Figure 2i
"""

#### importing packages ####
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt

def main():
    PVIdx=1
    LoadPath='../Data/DG_Default/sigmaEI30/sigmaIE30/TauNmda100/Fext1.8'
    for TrialID in range(100):
        print(TrialID)
        #### Hilbert transform for theta phase ####
        SpikeTimeE=np.load(LoadPath+'/SpikeTimeE_'+str(TrialID)+'.npz')['arr_0']
        PVs=np.load(LoadPath+'/PVs_'+str(TrialID)+'.npz')['arr_0']
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
        
        hist,binedges=np.histogram(SpikeInBump[:,0],bins=np.linspace(0,7000,1400,endpoint=False))
        sos = signal.butter(10, [4,12], 'bandpass', fs=200, output='sos')
        filthist=signal.sosfiltfilt(sos,hist)

        hilbertHist=signal.hilbert(filthist)
        hilbertPhase=np.arctan2(np.imag(hilbertHist),np.real(hilbertHist))
        TimeBins=(binedges[:-1]+binedges[1:])/2
        # fig,ax=plt.subplots()
        # ax.plot((binedges[:-1]+binedges[1:])/2,hist)
        # ax.plot((binedges[:-1]+binedges[1:])/2,filthist)
        # ax.plot((binedges[:-1]+binedges[1:])/2,np.abs(hilbertHist))
        # plt.show()
        # fig,ax=plt.subplots()
        # ax.plot((binedges[:-1]+binedges[1:])/2,hilbertPhase)
        # plt.show()

        #### theta phase of each neuron spike ####
        SpikeTimePhaseE=np.zeros((SpikeTimeE.shape[0],3))
        SpikeTimePhaseE[:,:2]=SpikeTimeE
        for i in range(SpikeTimeE.shape[0]):
            DeltaSpikeTime=np.abs(SpikeTimeE[i,0]-TimeBins)
            SpikeTimePhaseE[i,2]=hilbertPhase[np.argmin(DeltaSpikeTime)]

        np.savez_compressed(LoadPath+'/HilbertPhaseInTheta_'+str(TrialID)+'.npz',SH=hist,BE=binedges,FH=filthist,HTR=hilbertHist,HTP=hilbertPhase,TB=TimeBins,STP=SpikeTimePhaseE)

    #### calculate phaselocking index ####
    PVIdx=1
    VEIDs=np.load('../Data/DA_Figure2e-g/VEs.npz')['VEIds']
    ThetaE=np.linspace(0,360*(1024-1)/1024,1024)
    PhaseLockIndex=np.zeros((100,2))
    SpikeNum=np.zeros(100).astype(np.int64)
    for TrialID in range(100):
        # print(TrialID)
        SpikeTimePhaseE=np.load(LoadPath+'/HilbertPhaseInTheta_'+str(TrialID)+'.npz')['STP']
        SpikesForSingle_ID=np.where((np.abs(SpikeTimePhaseE[:,1]-ThetaE[VEIDs[TrialID]])<1e-3))[0]
        # print(SpikesForSingle_ID)

        # print(SpikesForSingle_ID.size)
        SpikeNum[TrialID]=SpikesForSingle_ID.size

        y=np.mean(np.sin(SpikeTimePhaseE[SpikesForSingle_ID,2]))
        x=np.mean(np.cos(SpikeTimePhaseE[SpikesForSingle_ID,2]))
        CenterPhase=np.arctan2(y,x)
        CenterHeight=np.sqrt(x**2+y**2)
        # print(CenterPhase,CenterHeight)

        PhaseLockIndex[TrialID,0]=CenterPhase
        PhaseLockIndex[TrialID,1]=CenterHeight

    np.savez_compressed('../Data/DA_Figure2i/PhaseLockIndex.npz',PLI=PhaseLockIndex)
    np.savez_compressed('../Data/DA_Figure2i/SpikeNum.npz',SpikeNum=SpikeNum)
    print(SpikeNum,np.mean(SpikeNum))

if __name__ == '__main__':
    main()
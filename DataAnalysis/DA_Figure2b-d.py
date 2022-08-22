# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 2021

@author: Tao WANG

Description: Analyzing data for Figure 2b-d
"""

#### importing packages ####
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt

def main():
    PVIdx=1
    Freq=np.zeros(513)
    Pxx_denE=np.zeros((513,100))
    Pxx_denIin=np.zeros((513,100))
    Pxx_denIout=np.zeros((513,100))

    for TrialID in range(100):
        LoadPath='../Data/DG_Default/sigmaEI30/sigmaIE30/TauNmda100/Fext1.8'
        PVs=np.load(LoadPath+'/PVs_'+str(TrialID)+'.npz')['arr_0']
        SpikeTimeE=np.load(LoadPath+'/SpikeTimeE_'+str(TrialID)+'.npz')['arr_0']
        SpikeTimeI=np.load(LoadPath+'/SpikeTimeI_'+str(TrialID)+'.npz')['arr_0']

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

        #### principal cells ####
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

        f, Pxx_den = signal.welch(hist, fs=200, window=signal.get_window('hamming',128), noverlap=64, nfft=1024)
        if TrialID==0: Freq=f
        Pxx_denE[:,TrialID]=Pxx_den
        print(f.shape,Pxx_denE.shape)

        #### interneurons ####
        SpikeInBump=np.array([[0,FinalPV[0]]])
        for i in range(SpikeTimeI.shape[0]):
            DeltaTime=np.abs(SpikeTimeI[i,0]-PVs[:,0])
            Idx=np.where(DeltaTime==np.min(DeltaTime))[0][0]
            if FinalPV[Idx]<60:
                if (360+(FinalPV[Idx]-60))<SpikeTimeI[i,1] or SpikeTimeI[i,1]<(FinalPV[Idx]+60):
                    SpikeInBump=np.append(SpikeInBump,np.array([SpikeTimeI[i,:]]),axis=0)
            elif FinalPV[Idx]<=300:
                if (FinalPV[Idx]-60)<SpikeTimeI[i,1] and SpikeTimeI[i,1]<(FinalPV[Idx]+60):
                    SpikeInBump=np.append(SpikeInBump,np.array([SpikeTimeI[i,:]]),axis=0)
            else:
                if (FinalPV[Idx]-60)<SpikeTimeI[i,1] or SpikeTimeI[i,1]<(FinalPV[Idx]+60-360):
                    SpikeInBump=np.append(SpikeInBump,np.array([SpikeTimeI[i,:]]),axis=0)
        
        histIn,binedgesIn=np.histogram(SpikeInBump[:,0],bins=np.linspace(0,7000,1400,endpoint=False))

        fIn, Pxx_denIn = signal.welch(histIn, fs=200, window=signal.get_window('hamming',128), noverlap=64, nfft=1024)
        Pxx_denIin[:,TrialID]=Pxx_denIn

        SpikeOutBump=np.array([[0,FinalPV[0]]])
        for i in range(SpikeTimeI.shape[0]):
            DeltaTime=np.abs(SpikeTimeI[i,0]-PVs[:,0])
            Idx=np.where(DeltaTime==np.min(DeltaTime))[0][0]
            ApartPV=FinalPV[Idx]+90
            if ApartPV>360: ApartPV-=360
            if ApartPV<60:
                if (360+(ApartPV-60))<SpikeTimeI[i,1] or SpikeTimeI[i,1]<(ApartPV+60):
                    SpikeOutBump=np.append(SpikeOutBump,np.array([SpikeTimeI[i,:]]),axis=0)
            elif ApartPV<=300:
                if (ApartPV-60)<SpikeTimeI[i,1] and SpikeTimeI[i,1]<(ApartPV+60):
                    SpikeOutBump=np.append(SpikeOutBump,np.array([SpikeTimeI[i,:]]),axis=0)
            else:
                if (ApartPV-60)<SpikeTimeI[i,1] or SpikeTimeI[i,1]<(ApartPV+60-360):
                    SpikeOutBump=np.append(SpikeOutBump,np.array([SpikeTimeI[i,:]]),axis=0)
        
        histOut,binedgesOut=np.histogram(SpikeOutBump[:,0],bins=np.linspace(0,7000,1400,endpoint=False))

        fOut, Pxx_denOut = signal.welch(histOut, fs=200, window=signal.get_window('hamming',128), noverlap=64, nfft=1024)
        Pxx_denIout[:,TrialID]=Pxx_denOut

    np.savez_compressed('../Data/DA_Figure2b-d/PSD.npz',Freq=Freq,PE=Pxx_denE,PII=Pxx_denIin,PIO=Pxx_denIout)

    SpikeFreq=Freq
    SpikePSD=Pxx_denE
    print(SpikePSD.shape)
    spikepeakf=np.zeros(100)
    spikepeakfintheta=np.zeros(100)
    for i in range(100):
        Mask=(4<=SpikeFreq)*(SpikeFreq<=12)
        spikepeakf[i]=SpikeFreq[np.where(SpikePSD[:,i]==np.max(SpikePSD[:,i]))[0][0]]
        spikepeakfintheta[i]=SpikeFreq[np.where(SpikePSD[:,i]*Mask==np.max(SpikePSD[:,i]*Mask))[0][0]]
        # print(spikepeakf[i])
        # print(spikepeakfintheta[i])

    np.savez_compressed('../Data/DA_Figure2b-d/spikepeakfs.npz',spikepeakf=spikepeakf,spikepeakfintheta=spikepeakfintheta)

if __name__ == '__main__':
    main()
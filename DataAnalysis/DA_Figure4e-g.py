# -*- coding: utf-8 -*-
"""
@author: Tao WANG, Ziqun WANG
Description: Analyzing data for Figure 4e-g
"""

#### importing packages ####
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
import sys

ie='30'#sys.argv[3]
Tau='100'#sys.argv[4]
DC='0'#sys.argv[2]
HW='0'#sys.argv[1]

def OneParaSet():
    PVIdx=1
    Freq=np.zeros(513)
    Pxx_denE=np.zeros((513,100))
    spikepeakf=np.zeros(100)
    spikepeakfintheta=np.zeros(100)
    Energyintheta=np.zeros(100)
    for TrialID in range(100):
        print(TrialID)
        SavePath='../Data/DG_Inhibition/sigmaIE'+ie+'/TauNmda'+Tau+'/DevtoCenter'+DC+'/HalfWidth'+HW
        PVs=np.load(SavePath+'/PVALs_'+str(TrialID)+'.npz')['arr_0']
        SpikeTimeE=np.load(SavePath+'/SpikeTimeE_'+str(TrialID)+'.npz')['arr_0']

        ####Frequency part####
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
        
        hist,binedges=np.histogram(SpikeInBump[:,0],bins=np.linspace(1000,7000,1200,endpoint=False))

        f, Pxx_den = signal.welch(hist, fs=200, window=signal.get_window('hamming',128), noverlap=64, nfft=1024)
        if TrialID==0: Freq=f
        Pxx_denE[:,TrialID]=Pxx_den
        # print(f.shape,Pxx_denE.shape)

        Mask=(4<=f)*(f<=12)
        spikepeakf[TrialID]=f[np.where(Pxx_den==np.max(Pxx_den))[0][0]]
        spikepeakfintheta[TrialID]=f[np.where(Pxx_den*Mask==np.max(Pxx_den*Mask))[0][0]]
        Energyintheta[TrialID]=np.sum(Pxx_den[Mask])/np.sum(Pxx_den)

    print(spikepeakf)
    print(Energyintheta)
    np.savez_compressed(SavePath+'/PSD.npz',Freq=Freq,PE=Pxx_denE)
    np.savez_compressed(SavePath+'/spikepeakfs.npz',spikepeakf=spikepeakf,spikepeakfintheta=spikepeakfintheta)
    np.savez_compressed(SavePath+'/energyfractionintheta.npz',EF=Energyintheta)

def VersusParam():
    DCs=[0,4,8,12,16,20,24]
    HWs=[0,4,8,12,16,20,24,28,32]
    MeanEnF=np.zeros(9)
    StdEnF=np.zeros(9)
    NumSpkInTheta=np.zeros(9)
    fig,ax=plt.subplots(figsize=(9,6))
    for i in range(9):
        hw=HWs[i]
        Path='../Data/DG_Inhibition/sigmaIE30/TauNmda100/DevtoCenter0/HalfWidth'+str(hw)

        EnF=np.load(Path+'/energyfractionintheta.npz')['EF']
        SpkF=np.load(Path+'/spikepeakfs.npz')['spikepeakf']

        MeanEnF[i]=np.mean(EnF)
        StdEnF[i]=np.std(EnF)
        NumSpkInTheta[i]=np.sum((4<=SpkF)*(SpkF<=12))
        ax.scatter(np.ones(100)*i,SpkF)
    plt.show()
    print(MeanEnF,StdEnF,NumSpkInTheta)
    np.savez_compressed('../Data/DA_Figure4e-g/HWs.npz',MeanEnF=MeanEnF,StdEnF=StdEnF,NumSpkInTheta=NumSpkInTheta)

    MeanEnF=np.zeros(7)
    StdEnF=np.zeros(7)
    NumSpkInTheta=np.zeros(7)
    fig,ax=plt.subplots(figsize=(9,6))
    for i in range(7):
        dc=DCs[i]
        Path='../Data/DG_Inhibition/sigmaIE30/TauNmda100/DevtoCenter'+str(dc)+'/HalfWidth16'
        EnF=np.load(Path+'/energyfractionintheta.npz')['EF']
        SpkF=np.load(Path+'/spikepeakfs.npz')['spikepeakf']
        MeanEnF[i]=np.mean(EnF)
        StdEnF[i]=np.std(EnF)
        NumSpkInTheta[i]=np.sum((4<=SpkF)*(SpkF<=12))
        ax.scatter(np.ones(100)*i,SpkF)
    plt.show()
    print(MeanEnF,StdEnF,NumSpkInTheta)
    np.savez_compressed('../Data/DA_Figure4e-g/DCs.npz',MeanEnF=MeanEnF,StdEnF=StdEnF,NumSpkInTheta=NumSpkInTheta)

if __name__ == '__main__':
    # OneParaSet()
    VersusParam()
# -*- coding: utf-8 -*-
"""
@author: Tao WANG, Ziqun WANG
Description: Analyzing data for Figure S1a-d
"""

#### importing packages ####
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt

GGabaE='0.577'#sys.argv[1]

def main():
    PVIdx=2
    Freq=np.zeros(513)
    Pxx_denE=np.zeros((513,1))
    for TrialID in range(1):
        print(TrialID)
        SavePath='../Data/DG_ThreeBumps/GGabaE'+GGabaE
        PVs=np.load(SavePath+'/PVs_'+str(TrialID)+'.npz')['arr_0']
        SpikeTimeE=np.load(SavePath+'/SpikeTimeE_'+str(TrialID)+'.npz')['arr_0']
        SpikeTimeI=np.load(SavePath+'/SpikeTimeI_'+str(TrialID)+'.npz')['arr_0']

        FinalPV1=PVs[:,1]/3
        FinalPV2=(PVs[:,1]+360)/3
        FinalPV3=(PVs[:,1]+2*360)/3
        for i in range(1,FinalPV1.size):
            DeltaAngle2=np.abs(FinalPV2[i]-FinalPV2[i-1])
            DeltaAngleA=np.abs(FinalPV2[i]-FinalPV1[i-1])
            DeltaAngleB=np.abs(FinalPV2[i]-FinalPV3[i-1])

            if DeltaAngleA<DeltaAngle2:####cross up bound
                FinalPV1[i],FinalPV2[i],FinalPV3[i]=FinalPV2[i],FinalPV3[i],FinalPV1[i]
                continue

            if DeltaAngleB<DeltaAngle2:####cross bottom bound
                FinalPV1[i],FinalPV2[i],FinalPV3[i]=FinalPV3[i],FinalPV1[i],FinalPV2[i]
                continue

        if PVIdx==1:
            FinalPV=FinalPV1
        elif PVIdx==2:
            FinalPV=FinalPV2
        else:
            FinalPV=FinalPV3

        SpikeInBump=np.array([[0,FinalPV[0]]])
        for i in range(SpikeTimeE.shape[0]):
            DeltaTime=np.abs(SpikeTimeE[i,0]-PVs[:,0])
            Idx=np.where(DeltaTime==np.min(DeltaTime))[0][0]
            if FinalPV[Idx]<30:
                if (360+(FinalPV[Idx]-30))<SpikeTimeE[i,1] or SpikeTimeE[i,1]<(FinalPV[Idx]+30):
                    SpikeInBump=np.append(SpikeInBump,np.array([SpikeTimeE[i,:]]),axis=0)
            elif FinalPV[Idx]<=330:
                if (FinalPV[Idx]-30)<SpikeTimeE[i,1] and SpikeTimeE[i,1]<(FinalPV[Idx]+30):
                    SpikeInBump=np.append(SpikeInBump,np.array([SpikeTimeE[i,:]]),axis=0)
            else:
                if (FinalPV[Idx]-30)<SpikeTimeE[i,1] or SpikeTimeE[i,1]<(FinalPV[Idx]+30-360):
                    SpikeInBump=np.append(SpikeInBump,np.array([SpikeTimeE[i,:]]),axis=0)
        
        hist,binedges=np.histogram(SpikeInBump[:,0],bins=np.linspace(0,7000,1400,endpoint=False))

        f, Pxx_den = signal.welch(hist, fs=200, window=signal.get_window('hamming',128), noverlap=64, nfft=1024)
        if TrialID==0: Freq=f
        Pxx_denE[:,TrialID]=Pxx_den

    np.savez_compressed('../Data/DA_FigureS1a-d/PSD.npz',Freq=Freq,PE=Pxx_denE)
    fig,ax=plt.subplots(figsize=(9,9))
    ax.plot(Freq[1:-1],np.mean(10*np.log10(Pxx_denE[1:-1,:]),axis=1))
    plt.savefig('../Data/DA_FigureS1a-d/PSD.png',fmt='PNG',dpi=150)
    plt.close()

if __name__ == '__main__':
    main()	
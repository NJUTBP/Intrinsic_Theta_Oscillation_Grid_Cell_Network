# -*- coding: utf-8 -*-
"""
@author: Tao WANG, Ziqun WANG
Description: Analyzing data for Figure 7a-e
"""

#### importing packages ####
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
import sys

def PopulationVector():
    FextList=np.array([1.0,1.2,1.4,1.6,1.8,2.0,2.2,2.4,2.6,2.8,3.0])
    FinalPVA=np.ones((7000,100,FextList.size))*np.nan
    FinalPVL=np.ones((7000,100,FextList.size))*np.nan
    for f in range(FextList.size):
        NormPVA=np.zeros((7000,100))
        NormPVL=np.zeros((7000,100))
        for TrialID in range(100):
            # print(TrialID)
            PVs=np.load('../Data/DG_Default/sigmaEI30/sigmaIE30/TauNmda100/Fext'+str(FextList[f])+'/PVs_'+str(TrialID)+'.npz')['arr_0']
            
            curtbin=0
            SumPV=np.zeros(2)
            SumCount=0
            (RowNum,ColNum)=PVs.shape
            # print(RowNum)
            for row in range(RowNum):
                tbin=int(PVs[row,0])
                if tbin>curtbin:
                    if SumCount>0:
                        x,y=(SumPV/SumCount).tolist()
                        NormAngle=180.0*np.arctan2(y,x)/np.pi
                        NormLength=np.sqrt(x**2+y**2)
                        if y<0:
                            NormAngle+=360.0
                        NormPVA[curtbin,TrialID]=NormAngle
                        NormPVL[curtbin,TrialID]=NormLength
                        
                    else:
                        NormPVA[curtbin,TrialID]=np.nan
                        NormPVL[curtbin,TrialID]=np.nan

                    curtbin=tbin
                    SumPV=np.zeros(2)
                    SumCount=0

                else:
                    SumPV+=np.array([PVs[row,2]*np.cos(PVs[row,1]*np.pi/180.0),PVs[row,2]*np.sin(PVs[row,1]*np.pi/180.0)])
                    SumCount+=1
            
            NormPVA[np.where(NormPVA[:,TrialID]==0.0),TrialID]=np.nan
            NormPVL[np.where(NormPVL[:,TrialID]==0.0),TrialID]=np.nan
        
        FinalPVL[:,:,f]=NormPVL

        UpNormPVA=NormPVA+360
        DownNormPVA=NormPVA-360
        UpUpNormPVA=NormPVA+720
        DownDownNormPVA=NormPVA-720
        DownDownDownNormPVA=NormPVA-1080
        for i in range(100):
            EffAngle=0
            for j in range(7000):
                DeltaA1=np.abs(NormPVA[j,i]-EffAngle)
                DeltaA2=np.abs(UpNormPVA[j,i]-EffAngle)
                DeltaA3=np.abs(DownNormPVA[j,i]-EffAngle)
                DeltaA4=np.abs(UpUpNormPVA[j,i]-EffAngle)
                DeltaA5=np.abs(DownDownNormPVA[j,i]-EffAngle)
                DeltaA6=np.abs(DownDownDownNormPVA[j,i]-EffAngle)
                minDelta=np.min([DeltaA1,DeltaA2,DeltaA3,DeltaA4,DeltaA5,DeltaA6])
                if np.isnan(NormPVA[j,i]):
                    continue
                elif minDelta==DeltaA1:
                    FinalPVA[j,i,f]=NormPVA[j,i]
                    EffAngle=NormPVA[j,i]
                elif minDelta==DeltaA2:
                    FinalPVA[j,i,f]=UpNormPVA[j,i]
                    EffAngle=UpNormPVA[j,i]
                elif minDelta==DeltaA3:
                    FinalPVA[j,i,f]=DownNormPVA[j,i]
                    EffAngle=DownNormPVA[j,i]
                elif minDelta==DeltaA4:
                    FinalPVA[j,i,f]=UpUpNormPVA[j,i]
                    EffAngle=UpUpNormPVA[j,i]
                elif minDelta==DeltaA5:
                    FinalPVA[j,i,f]=DownDownNormPVA[j,i]
                    EffAngle=DownDownNormPVA[j,i]
                else:
                    FinalPVA[j,i,f]=DownDownDownNormPVA[j,i]
                    EffAngle=DownDownDownNormPVA[j,i]

        for i in range(100):
            nans, x= np.isnan(FinalPVA[:,i,f]), lambda z: z.nonzero()[0]
            print(i,np.sum(nans))
            # print(x(nans), x(~nans), FinalPVA[~nans,i,f])
            FinalPVA[nans,i,f]=np.interp(x(nans), x(~nans), FinalPVA[~nans,i,f])
            FinalPVA[:,i,f]=FinalPVA[:,i,f]-FinalPVA[750,i,f]

    np.savez_compressed('../Data/DA_Figure7a-e/AllPVs.npz',PVA=FinalPVA,
                                                            PVL=FinalPVL)

def ThetaRhythm():
    FextList=np.array([1.0,1.2,1.4,1.6,1.8,2.0,2.2,2.4,2.6,2.8,3.0])
    PVIdx=1
    NE=1024
    Energyintheta=np.zeros((100,FextList.size))
    PSD=np.zeros((513,100,FextList.size))
    #### calculate the theta power ####
    for i in range(FextList.size):
        for j in range(100):
            if i==0 and j==0:
                Freq=np.load('../Data/DG_Default/sigmaEI30/sigmaIE30/TauNmda100/Fext'+str(FextList[i])+'/PSD_'+str(j)+'.npz')['Freq']

            if FextList[i] in [1.0,1.2,1.4]: 
                SpikeTimeE=np.load('../Data/DG_Default/sigmaEI30/sigmaIE30/TauNmda100/Fext'+str(FextList[i])+'/SpikeTimeE_'+str(j)+'.npz')['arr_0']
                hist,binedges=np.histogram(SpikeTimeE[:,0],bins=np.linspace(0,7000,1400,endpoint=False))
                f, Pxx_den = signal.welch(hist, fs=200, window=signal.get_window('hamming',128), noverlap=64, nfft=1024)
                np.savez_compressed('../Data/DG_Default/sigmaEI30/sigmaIE30/TauNmda100/Fext'+str(FextList[i])+'/AllNeuPSD_'+str(j)+'.npz',Freq=f,PE=Pxx_den)
                PSD[:,j,i]=Pxx_den
            else:
                PSD[:,j,i]=np.load('../Data/DG_Default/sigmaEI30/sigmaIE30/TauNmda100/Fext'+str(FextList[i])+'/PSD_'+str(j)+'.npz')['PE']
            
            Mask=(4<=Freq)*(Freq<=12)
            Energyintheta[j,i]=np.sum(PSD[Mask,j,i])/np.sum(PSD[:,j,i])

    #### calculate theta frequency ####
    AllPeakInTheta_AutoCor=np.zeros((100,NE,FextList.size))*np.nan
    for n in range(FextList.size):
        print(FextList[n])
        PeakInTheta_AutoCor=np.zeros((100,NE))*np.nan
        ThetaE=np.linspace(0,360*(NE-1)/NE,NE)
        LoadPath='../Data/DG_Default/sigmaEI30/sigmaIE30/TauNmda100/Fext'+str(FextList[n])
        for TrialID in range(100):
            print(TrialID)
            #### frequency by spiketrain auto correlation ####
            SpikeTimeE=np.load(LoadPath+'/SpikeTimeE_'+str(TrialID)+'.npz')['arr_0']
            NeuronLabelsInThisTrial=np.unique(SpikeTimeE[:,1])
            ACbinedges=np.linspace(0,1,201)
            ACbins=(ACbinedges[:-1]+ACbinedges[1:])/2
            Mask=((1/12)<ACbins)*(ACbins<(1/4))
            for i in range(NeuronLabelsInThisTrial.size):
                NeuronID=np.where(np.abs(ThetaE-NeuronLabelsInThisTrial[i])<1e-3)
                SpikeTimeForSingleE=SpikeTimeE[np.where(np.abs(SpikeTimeE[:,1]-NeuronLabelsInThisTrial[i])<1e-3)[0],0]

                hist,binedges=np.histogram(SpikeTimeForSingleE,bins=np.linspace(0,7000,1400,endpoint=False))

                AChist=np.zeros(200)
                AChist[0]=np.sum(hist*hist)
                for delay in range(1,200):
                    AChist[delay]=np.sum(hist[delay:]*hist[:-delay])

                MaskAChist=AChist*Mask
                # print(i,1/ACbins[np.where((MaskAChist)==np.max((MaskAChist)))[0][0]])
                if np.sum(MaskAChist)>0:
                    PeakInTheta_AutoCor[TrialID,NeuronID]=1/ACbins[np.where((MaskAChist)==np.max((MaskAChist)))[0][0]]
        
        AllPeakInTheta_AutoCor[:,:,n]=PeakInTheta_AutoCor
    
    np.savez_compressed('../Data/DA_Figure7a-e/ThetaRhythm.npz',Energyintheta=Energyintheta,
                                                                AllPeakInTheta_AutoCor=AllPeakInTheta_AutoCor)

if __name__ == '__main__':
    PopulationVector()
    ThetaRhythm()
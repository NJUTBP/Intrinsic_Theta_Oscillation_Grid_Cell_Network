# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 2021

@author: Tao WANG

Description: Analyzing data for Figure S2 and S6
"""

#### importing packages ####
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt

def mainforallCR_v2():
    CRs=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    T=30000.0
    PVAofAllCR=np.zeros((int(T),100,10))
    PSDofAllCR=np.zeros((513,4,100,10))
    ThPofAllCR=np.zeros((4,10))
    for CRid in range(10):
        CR=str(CRs[CRid])
        print(CR)
        LoadPath='../Data/DG_CoherentBack/CR'+CR

        #### PV Part ####
        NormPVA=np.zeros((int(T),100))
        NormPVL=np.zeros((int(T),100))

        for TrialID in range(100):
            print(TrialID)
            PVs=np.load(LoadPath+'/PVs_'+str(TrialID)+'.npz')['arr_0']

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
            
        UpNormPVA=NormPVA+360
        DownNormPVA=NormPVA-360
        UpUpNormPVA=NormPVA+720
        DownDownNormPVA=NormPVA-720
        DownDownDownNormPVA=NormPVA-1080
        FinalPVA=np.ones((int(T),100))*np.nan
        for i in range(100):
            EffAngle=0
            for j in range(int(T)):
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
                    FinalPVA[j,i]=NormPVA[j,i]
                    EffAngle=NormPVA[j,i]
                elif minDelta==DeltaA2:
                    FinalPVA[j,i]=UpNormPVA[j,i]
                    EffAngle=UpNormPVA[j,i]
                elif minDelta==DeltaA3:
                    FinalPVA[j,i]=DownNormPVA[j,i]
                    EffAngle=DownNormPVA[j,i]
                elif minDelta==DeltaA4:
                    FinalPVA[j,i]=UpUpNormPVA[j,i]
                    EffAngle=UpUpNormPVA[j,i]
                elif minDelta==DeltaA5:
                    FinalPVA[j,i]=DownDownNormPVA[j,i]
                    EffAngle=DownDownNormPVA[j,i]
                else:
                    FinalPVA[j,i]=DownDownDownNormPVA[j,i]
                    EffAngle=DownDownDownNormPVA[j,i]

        for i in range(100):
            nans, x= np.isnan(FinalPVA[:,i]), lambda z: z.nonzero()[0]
            print(i,np.sum(nans))
            # print(x(nans), x(~nans), FinalPVA[~nans,i])
            FinalPVA[nans,i]=np.interp(x(nans), x(~nans), FinalPVA[~nans,i])
            FinalPVA[:,i]=FinalPVA[:,i]-FinalPVA[750,i]

        np.save(LoadPath+'/AllPVA.npy',FinalPVA)
        PVAofAllCR[:,:,CRid]=FinalPVA

        #### Theta Part ####
        PVIdx=1
        Freq=np.zeros(513)
        Pxx_denE=np.zeros((513,4,100))
        for TrialID in range(100):
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

            # print(FinalPV.shape,SpikeTimeE.shape)
            SpikeInBump=np.array([[0,FinalPV[0]]])
            SpikeTimeE[:,1]
            SpikeInBumpFlag=(FinalPV[1:]<60)*(((360+(FinalPV[1:]-60))<SpikeTimeE[:,1])+(SpikeTimeE[:,1]<(FinalPV[1:]+60)))+(60<=FinalPV[1:])*(FinalPV[1:]<=300)*((FinalPV[1:]-60)<SpikeTimeE[:,1])*(SpikeTimeE[:,1]<(FinalPV[1:]+60))+(FinalPV[1:]>300)*(((FinalPV[1:]-60)<SpikeTimeE[:,1])+(SpikeTimeE[:,1]<(FinalPV[1:]+60-360)))
            SpikeInBump=SpikeTimeE[SpikeInBumpFlag,:]
            hist,binedges=np.histogram(SpikeInBump[:,0],bins=np.linspace(0,int(T),int(T/5+1),endpoint=False))
            # print(hist.shape)

            hist_1=hist[150:1550]#[750:7750]ms
            hist_2=hist[1550:2950]#[7750:14750]ms
            hist_3=hist[2950:4350]#[14750:21750]ms
            hist_4=hist[4350:5750]#[21750:28750]ms
            # fig,ax=plt.subplots()
            # ax.plot(np.arange(750,7750,5),hist_1)
            # ax.plot(np.arange(7750,14750,5),hist_2)
            # ax.plot(np.arange(14750,21750,5),hist_3)
            # ax.plot(np.arange(21750,28750,5),hist_4)
            # plt.show()

            f, Pxx_den_1 = signal.welch(hist_1, fs=200, window=signal.get_window('hamming',128), noverlap=64, nfft=1024)
            f, Pxx_den_2 = signal.welch(hist_2, fs=200, window=signal.get_window('hamming',128), noverlap=64, nfft=1024)
            f, Pxx_den_3 = signal.welch(hist_3, fs=200, window=signal.get_window('hamming',128), noverlap=64, nfft=1024)
            f, Pxx_den_4 = signal.welch(hist_4, fs=200, window=signal.get_window('hamming',128), noverlap=64, nfft=1024)
            
            if TrialID==0: Freq=f
            Pxx_denE[:,0,TrialID]=Pxx_den_1
            Pxx_denE[:,1,TrialID]=Pxx_den_2
            Pxx_denE[:,2,TrialID]=Pxx_den_3
            Pxx_denE[:,3,TrialID]=Pxx_den_4
            # print(f.shape,Pxx_denE.shape)
        
        MeanPSD=np.mean(Pxx_den,axis=2)
        Mask=np.array([(4<=Freq)*(Freq<=12)]*4).T
        ThetaPower=np.sum(MeanPSD*Mask,axis=0)/np.sum(MeanPSD,axis=0)
        ThPofAllCR[:,CRid]=ThetaPower

        np.savez_compressed(LoadPath+'/AllPSD_4parts.npz',Freq=f,PSD=Pxx_denE)
        PSDofAllCR[:,:,:,CRid]=Pxx_denE
        
        # fig,ax=plt.subplots()
        # ax.plot(f,Pxx_denE[:,0,0])
        # ax.plot(f,Pxx_denE[:,1,0])
        # ax.plot(f,Pxx_denE[:,2,0])
        # ax.plot(f,Pxx_denE[:,3,0])
        # plt.show()

    np.savez_compressed('../Data/DG_CoherentBack/PVA_PSD_4parts.npz',CRs=CRs,PVA=PVAofAllCR,Freq=f,PSD=PSDofAllCR,ThetaPower=ThPofAllCR)

if __name__ == '__main__':
    mainforallCR_v2()
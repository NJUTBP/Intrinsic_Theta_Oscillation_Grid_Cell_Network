# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 2021

@author: Tao WANG

Description: Analyzing data for Figure 5a-f
"""
#### importing packages ####
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
import sys

def PeakInTheta():
    PVIdx=1
    NE=1024
    PeakInTheta_SimpleS=np.zeros(100)
    PeakInTheta_FilterS=np.zeros(100)
    PeakInTheta_AutoCor=np.zeros((100,NE))*np.nan
    ThetaE=np.linspace(0,360*(NE-1)/NE,NE)
    LoadPath='../Data/DG_OsciBack/FextFreq'+sys.argv[1]
    for TrialID in range(100):
        print(TrialID)
        
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
        f, Pxx_den = signal.welch(hist, fs=200, window=signal.get_window('hamming',128), noverlap=64, nfft=1024)
        Mask=(4<=f)*(f<=12)

        # sos = signal.butter(10, [4,12], 'bandpass', fs=200, output='sos')
        sos = signal.butter(10, [0.04,0.12], 'bandpass', output='sos')
        filthist=signal.sosfilt(sos,hist)

        ffilt, Pxx_denfilt = signal.welch(filthist, fs=200, window=signal.get_window('hamming',128), noverlap=64, nfft=1024)

        # fig,ax=plt.subplots()
        # ax.plot(f,Pxx_den)
        # ax.plot(ffilt,Pxx_denfilt)
        # plt.show()

        print(f[np.where(Pxx_den*Mask==np.max(Pxx_den*Mask))[0][0]],ffilt[np.where(Pxx_denfilt==np.max(Pxx_denfilt))[0][0]])
        PeakInTheta_SimpleS[TrialID]=f[np.where(Pxx_den*Mask==np.max(Pxx_den*Mask))[0][0]]
        PeakInTheta_FilterS[TrialID]=ffilt[np.where(Pxx_denfilt==np.max(Pxx_denfilt))[0][0]]

        #### theta frequency by spiketrain auto correlation ####
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

    np.savez_compressed(LoadPath+'/PeaksInTheta.npz',PeakInTheta_SimpleS=PeakInTheta_SimpleS,
                                                    PeakInTheta_FilterS=PeakInTheta_FilterS,
                                                    PeakInTheta_AutoCor=PeakInTheta_AutoCor)

def HilbertTransform():
    PVIdx=1
    LoadPath='../Data/DG_OsciBack/FextFreq'+sys.argv[1]
    for TrialID in range(100):
        print(TrialID)
        #### hilbert transformation for theta phase ####
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
        # sos = signal.butter(10, [4,12], 'bandpass', fs=200, output='sos')
        sos = signal.butter(10, [0.04,0.12], 'bandpass', output='sos')
        filthist=signal.sosfiltfilt(sos,hist)

        hilbertHist=signal.hilbert(filthist)
        hilbertPhase=np.arctan2(np.imag(hilbertHist),np.real(hilbertHist))
        TimeBins=(binedges[:-1]+binedges[1:])/2

        InputRhythm=np.cos(2*np.pi*TimeBins*float(sys.argv[1])/1000)

        # fig,ax=plt.subplots()
        # ax.plot(TimeBins,InputRhythm)
        # ax.plot(TimeBins,filthist)
        # plt.show()
        # fig,ax=plt.subplots()
        # ax.plot((binedges[:-1]+binedges[1:])/2,hilbertPhase)
        # plt.show()

        #### theta phase of each spike relative to LFP theta ####
        SpikeTimePhaseE=np.zeros((SpikeTimeE.shape[0],4))
        SpikeTimePhaseE[:,:2]=SpikeTimeE
        for i in range(SpikeTimeE.shape[0]):
            DeltaSpikeTime=np.abs(SpikeTimeE[i,0]-TimeBins)
            SpikeTimePhaseE[i,2]=hilbertPhase[np.argmin(DeltaSpikeTime)]
            SpikeTimePhaseE[i,3]=2*np.pi*SpikeTimeE[i,0]*float(sys.argv[1])/1000

        np.savez_compressed(LoadPath+'/HilbertPhaseInTheta_'+str(TrialID)+'.npz',SH=hist,BE=binedges,FH=filthist,HTR=hilbertHist,HTP=hilbertPhase,IR=InputRhythm,TB=TimeBins,STP=SpikeTimePhaseE)

def PSDAbstract():
    FFList=[2,4,5,6,7,8,9,10,11,12,14,16,18,20,24,28,32,40,48]
    PSD=np.zeros((513,100,19))
    Freq=np.zeros(513)
    spikepeakf=np.zeros((100,19))
    spikepeakfintheta=np.zeros((100,19))
    Energyintheta=np.zeros((100,19))
    SecondaryPeakf=np.zeros((100,19))
    TertiaryPeakf=np.zeros((100,19))
    for i in range(19):
        for j in range(100):
            if i==0 and j==0:
                Freq=np.load('../Data/DG_OsciBack/FextFreq'+str(FFList[i])+'/PSD_'+str(j)+'.npz')['Freq']
            PSD[:,j,i]=np.load('../Data/DG_OsciBack/FextFreq'+str(FFList[i])+'/PSD_'+str(j)+'.npz')['PE']
            
            Mask=(4<=Freq)*(Freq<=12)
            spikepeakf[j,i]=Freq[np.where(PSD[:,j,i]==np.max(PSD[:,j,i]))[0][0]]
            spikepeakfintheta[j,i]=Freq[np.where(PSD[:,j,i]*Mask==np.max(PSD[:,j,i]*Mask))[0][0]]
            Energyintheta[j,i]=np.sum(PSD[Mask,j,i])/np.sum(PSD[:,j,i])

            LocalMaxIdx=np.where((PSD[:-2,j,i]<PSD[1:-1,j,i])*(PSD[1:-1,j,i]>PSD[2:,j,i]))[0]+1
            ArgLocalMaxIdx=LocalMaxIdx[np.argsort(PSD[LocalMaxIdx,j,i])]
            if j==0:
                print(Freq[ArgLocalMaxIdx])#,PSD[ArgLocalMaxIdx,j,i])
            if spikepeakf[j,i]!=Freq[ArgLocalMaxIdx[-1]]:
                print('Error!!!')
            else:
                SecondaryPeakf[j,i]=Freq[ArgLocalMaxIdx[-2]]
                TertiaryPeakf[j,i]=Freq[ArgLocalMaxIdx[-3]]

    np.savez_compressed('../Data/DA_Figure5a-f/AllData.npz',FFList=FFList,
                                                            PSD=PSD,
                                                            Freq=Freq,
                                                            spikepeakf=spikepeakf,
                                                            spikepeakfintheta=spikepeakfintheta,
                                                            Energyintheta=Energyintheta,
                                                            SecondaryPeakf=SecondaryPeakf,
                                                            TertiaryPeakf=TertiaryPeakf)

def SCvsInput():
    PVIdx=1
    FFList=[2,4,5,6,7,8,9,10,11,12,14,16,18,20,24,28,32,40,48]
    NE=1024
    AllPhaseLockIndex=np.zeros((100,6,19))
    AllPeakInTheta_SimpleS=np.zeros((100,19))
    AllPeakInTheta_FilterS=np.zeros((100,19))
    AllPeakInTheta_AutoCor=np.zeros((100,NE,19))*np.nan
    for ff in range(19):
        print(ff)
        LoadPath='../Data/DG_OsciBack/FextFreq'+str(FFList[ff])
        ThetaE=np.linspace(0,360*(NE-1)/NE,NE)
        PhaseLockIndex=np.zeros((100,6))
        for TrialID in range(100):
            # print(TrialID)
            HilbertPhase=np.load(LoadPath+'/HilbertPhaseInTheta_'+str(TrialID)+'.npz')['HTP']
            FiltResult=np.load(LoadPath+'/HilbertPhaseInTheta_'+str(TrialID)+'.npz')['FH']
            InputRhythm=np.load(LoadPath+'/HilbertPhaseInTheta_'+str(TrialID)+'.npz')['IR']
            TimeBins=np.load(LoadPath+'/HilbertPhaseInTheta_'+str(TrialID)+'.npz')['TB']
            InputPhase=2*np.pi*TimeBins*FFList[ff]/1000
            InputPhase=InputPhase-2*np.pi*(InputPhase/2/np.pi).astype(np.int64)-np.pi####这里不应该减，会导致后面作图差半个相位
            SpikeTimePhaseE=np.load(LoadPath+'/HilbertPhaseInTheta_'+str(TrialID)+'.npz')['STP']

            ####calculate the difference between rhythms of input and spike count ####
            DeltaPhase=HilbertPhase-InputPhase
            DeltaPhase=DeltaPhase+(DeltaPhase<0)*2*np.pi
            y=np.mean(np.sin(DeltaPhase))
            x=np.mean(np.cos(DeltaPhase))
            CenterPhase=np.arctan2(y,x)
            CenterHeight=np.sqrt(x**2+y**2)
            y1=np.mean(np.sin(DeltaPhase[:int(DeltaPhase.size/2)]))
            x1=np.mean(np.cos(DeltaPhase[:int(DeltaPhase.size/2)]))
            CenterPhase1=np.arctan2(y1,x1)
            CenterHeight1=np.sqrt(x1**2+y1**2)
            y2=np.mean(np.sin(DeltaPhase[int(DeltaPhase.size/2):]))
            x2=np.mean(np.cos(DeltaPhase[int(DeltaPhase.size/2):]))
            CenterPhase2=np.arctan2(y2,x2)
            CenterHeight2=np.sqrt(x2**2+y2**2)
            # print(CenterPhase,CenterHeight,CenterPhase1,CenterHeight1,CenterPhase2,CenterHeight2)
            PhaseLockIndex[TrialID,:]=np.array([CenterPhase,CenterHeight,CenterPhase1,CenterHeight1,CenterPhase2,CenterHeight2])
        
        np.savez_compressed(LoadPath+'/PhaseLockIndex.npz',PhaseLockIndex)
        AllPhaseLockIndex[:,:,ff]=PhaseLockIndex

        AllPeakInTheta_SimpleS[:,ff]=np.load(LoadPath+'_PeaksInTheta.npz')['PeakInTheta_SimpleS']
        AllPeakInTheta_FilterS[:,ff]=np.load(LoadPath+'_PeaksInTheta.npz')['PeakInTheta_FilterS']
        AllPeakInTheta_AutoCor[:,:,ff]=np.load(LoadPath+'_PeaksInTheta.npz')['PeakInTheta_AutoCor']

    np.savez_compressed('../Data/DA_Figure5a-f/AllPeaksInTheta.npz',AllPeakInTheta_SimpleS=AllPeakInTheta_SimpleS,
                                                                    AllPeakInTheta_FilterS=AllPeakInTheta_FilterS,
                                                                    AllPeakInTheta_AutoCor=AllPeakInTheta_AutoCor)
    np.savez_compressed('../Data/DA_Figure5a-f/AllPhaseLockIndex.npz',AllPhaseLockIndex=AllPhaseLockIndex)
        
def SpikePhase():
    PVIdx=1
    FFList=[2,4,5,6,7,8,9,10,11,12,14,16,18,20,24,28,32,40,48]
    NE=1024
    AllPhaseLockIndex=np.zeros((100,4,19))
    for ff in range(19):
        print(ff)
        LoadPath='../Data/DG_OsciBack/FextFreq'+str(FFList[ff])
        ThetaE=np.linspace(0,360*(NE-1)/NE,NE)
        for TrialID in range(100):
            ProfE=np.load(LoadPath+'/Profile_'+str(TrialID)+'.npz')['arr_0']
            SpikeTimePhaseE=np.load(LoadPath+'/HilbertPhaseInTheta_'+str(TrialID)+'.npz')['STP']

            SpikeTimePhaseForSingleE_ID=np.where(np.abs(SpikeTimePhaseE[:,1]-ThetaE[np.argmax(ProfE)])<1e-3)[0]
            
            y=np.mean(np.sin(SpikeTimePhaseE[SpikeTimePhaseForSingleE_ID,2]))
            x=np.mean(np.cos(SpikeTimePhaseE[SpikeTimePhaseForSingleE_ID,2]))
            CenterPhase=np.arctan2(y,x)
            CenterHeight=np.sqrt(x**2+y**2)
            # print(CenterPhase,CenterHeight)

            AllPhaseLockIndex[TrialID,0,ff]=CenterPhase
            AllPhaseLockIndex[TrialID,1,ff]=CenterHeight

            y=np.mean(np.sin(SpikeTimePhaseE[SpikeTimePhaseForSingleE_ID,3]))
            x=np.mean(np.cos(SpikeTimePhaseE[SpikeTimePhaseForSingleE_ID,3]))
            CenterPhase=np.arctan2(y,x)
            CenterHeight=np.sqrt(x**2+y**2)
            # print(CenterPhase,CenterHeight)

            AllPhaseLockIndex[TrialID,2,ff]=CenterPhase
            AllPhaseLockIndex[TrialID,3,ff]=CenterHeight

    np.savez_compressed('../Data/DA_Figure5a-f/SpikePhaseLockIndex.npz',AllPhaseLockIndex=AllPhaseLockIndex)
        
def Resonance():
    FFList=np.load('../Data/DA_Figure5a-f/AllData.npz')['FFList']
    PSD=np.load('../Data/DA_Figure5a-f/AllData.npz')['PSD']
    Freq=np.load('../Data/DA_Figure5a-f/AllData.npz')['Freq']
    spikepeakf=np.load('../Data/DA_Figure5a-f/AllData.npz')['spikepeakf']
    print(FFList.shape)
    print(PSD.shape)
    print(Freq.shape)
    print(spikepeakf.shape)
    PeakEnergy=np.ones((100,19))*np.nan
    for ff in range(19):
        for TrialID in range(100):
            if spikepeakf[TrialID,ff]>2:
                bottomboundID=np.where(np.abs(Freq-spikepeakf[TrialID,ff]+2)<0.1)[0][0]
                upperboundID=np.where(np.abs(Freq-spikepeakf[TrialID,ff]-2)<0.1)[0][0]
            else:
                bottomboundID=0
                upperboundID=np.where(np.abs(Freq-spikepeakf[TrialID,ff]-2)<0.1)[0][0]
            PeakEnergy[TrialID,ff]=np.sum(PSD[bottomboundID:upperboundID,TrialID,ff])/np.sum(PSD[:,TrialID,ff])

    # fig,ax=plt.subplots()
    # ax.plot(FFList,np.nanmean(PeakEnergy,axis=0))
    # plt.savefig('../Data/DA_Figure5a-f/Resonance.png',fmt='PNG',dpi=150)
    # plt.show()
    
    np.savez_compressed('../Data/DA_Figure5a-f/Resonance.npz',PeakEnergy=PeakEnergy)
        
if __name__ == '__main__':
    # PeakInTheta()
    # HilbertTransform()
    PSDAbstract()
    SCvsInput()
    SpikePhase()
    Resonance()
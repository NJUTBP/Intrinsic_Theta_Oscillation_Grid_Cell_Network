# -*- coding: utf-8 -*-
"""
@author: Tao WANG, Ziqun WANG
Description: Analyzing data for Figure 3h
"""

#### importing packages ####
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt

def main():
    PVIdx=1
    FiltPeakTheta=np.zeros(100)
    VEs=np.load('../Data/DA_Figure3e-g/VEs.npz')['VEs']
    VEIDs=np.load('../Data/DA_Figure3e-g/VEs.npz')['VEIds']
    print(VEIDs)
    # return 0
    NE=1024
    PeakInTheta_SimpleS=np.zeros(100)
    PeakInTheta_FilterS=np.zeros(100)
    PeakInTheta_SimpleV=np.zeros(100)
    PeakInTheta_FilterV=np.zeros(100)
    PeakInTheta_AutoCor=np.zeros((100,NE))*np.nan
    PeakInTheta_AutoCor_Draw=np.zeros(100)
    ThetaE=np.linspace(0,360*(NE-1)/NE,NE)
    for TrialID in range(100):
        print(TrialID)
        #### loading spike count ####
        LoadPath='../Data/DG_Default/sigmaEI30/sigmaIE30/TauNmda100/Fext1.8'
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

        sos = signal.butter(10, [4,12], 'bandpass', fs=200, output='sos')
        filthist=signal.sosfilt(sos,hist)
        ffilt, Pxx_denfilt = signal.welch(filthist, fs=200, window=signal.get_window('hamming',128), noverlap=64, nfft=1024)

        print(f[np.where(Pxx_den*Mask==np.max(Pxx_den*Mask))[0][0]],ffilt[np.where(Pxx_denfilt==np.max(Pxx_denfilt))[0][0]])
        PeakInTheta_SimpleS[TrialID]=f[np.where(Pxx_den*Mask==np.max(Pxx_den*Mask))[0][0]]
        PeakInTheta_FilterS[TrialID]=ffilt[np.where(Pxx_denfilt==np.max(Pxx_denfilt))[0][0]]

        #### loading membrane potential ####
        VE=VEs[:,TrialID]
        f, Pxx_den = signal.welch(np.mean(VE.reshape((400,25)),axis=1), fs=200, window=signal.get_window('hamming',128), noverlap=64, nfft=1024)
        Mask=(4<=f)*(f<=12)
        # peakfintheta[TrialID]=f[np.where(Pxx_den*Mask==np.max(Pxx_den*Mask))[0][0]]

        sos = signal.butter(10, [4,12], 'bandpass', fs=200, output='sos')
        filtVE=signal.sosfilt(sos,np.mean(VE.reshape((400,25)),axis=1))
        ffilt, Pxx_denfilt = signal.welch(filtVE, fs=200, window=signal.get_window('hamming',128), noverlap=64, nfft=1024)

        print(f[np.where(Pxx_den*Mask==np.max(Pxx_den*Mask))[0][0]],ffilt[np.where(Pxx_denfilt==np.max(Pxx_denfilt))[0][0]])
        PeakInTheta_SimpleV[TrialID]=f[np.where(Pxx_den*Mask==np.max(Pxx_den*Mask))[0][0]]
        PeakInTheta_FilterV[TrialID]=ffilt[np.where(Pxx_denfilt==np.max(Pxx_denfilt))[0][0]]

        #### calculating spiketrain auto correlation ####
        NeuronLabelsInThisTrial=np.unique(SpikeTimeE[:,1])
        ACbinedges=np.linspace(0,1,201)
        ACbins=(ACbinedges[:-1]+ACbinedges[1:])/2
        Mask=((1/12)<=ACbins)*(ACbins<=(1/4))
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
            if NeuronID==VEIDs[TrialID]:
                PeakInTheta_AutoCor_Draw[TrialID]=PeakInTheta_AutoCor[TrialID,NeuronID]
                # fig,ax=plt.subplots()
                # ax.plot(AChist)
                # plt.show()
                # np.savez_compressed('../Data/DA_Figure2h/ACHist.npz',AChist=AChist)

    np.savez_compressed('../Data/DA_Figure3h/PeaksInTheta.npz',PeakInTheta_SimpleS=PeakInTheta_SimpleS,
                                                        PeakInTheta_FilterS=PeakInTheta_FilterS,
                                                        PeakInTheta_SimpleV=PeakInTheta_SimpleV,
                                                        PeakInTheta_FilterV=PeakInTheta_FilterV,
                                                        PeakInTheta_AutoCor=PeakInTheta_AutoCor,
                                                        PeakInTheta_AutoCor_ToV=PeakInTheta_AutoCor_Draw)

if __name__ == '__main__':
    main()
# -*- coding: utf-8 -*-
"""
@author: Tao WANG, Ziqun WANG
Description: Analyzing data for Figure 3e-g
"""

#### importing packages ####
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt

def main():
    NE,NI=1024,256
    ThetaE=np.linspace(0,360*(NE-1)/NE,NE)
    ThetaI=np.linspace(0,360*(NI-1)/NI,NI)
    # TrialID=8
    PVIdx=1
    VEs=np.zeros((10000,100))
    VEIds=np.zeros(100).astype(np.int64)
    Pxxs=np.zeros((513,100))
    peakf=np.zeros(100)
    peakfintheta=np.zeros(100)
    
    for TrialID in range(100):
        LoadPath='../Data/DG_Default/sigmaEI30/sigmaIE30/TauNmda100/Fext1.8'
        SpikeTimeE=np.load(LoadPath+'/SpikeTimeE_'+str(TrialID)+'.npz')['arr_0']
        SpikeTimeI=np.load(LoadPath+'/SpikeTimeI_'+str(TrialID)+'.npz')['arr_0']
        ProfE=np.load(LoadPath+'/Profile_'+str(TrialID)+'.npz')['arr_0']
        SingleBVE=np.load(LoadPath+'/SingleBVE_'+str(TrialID)+'.npz')['arr_0']
        SingleBVI=np.load(LoadPath+'/SingleBVI_'+str(TrialID)+'.npz')['arr_0']
        SingleVE=np.load(LoadPath+'/SingleVE_'+str(TrialID)+'.npz')['arr_0']
        SingleVI=np.load(LoadPath+'/SingleVI_'+str(TrialID)+'.npz')['arr_0']
        SingleUVE=np.load(LoadPath+'/SingleUVE_'+str(TrialID)+'.npz')['arr_0']
        SingleUVI=np.load(LoadPath+'/SingleUVI_'+str(TrialID)+'.npz')['arr_0']
        # print(SingleVE.shape)
        print(TrialID)
        
        ThetaEB=np.array([364,341,313,284,256,228,199,171,148])
        ThetaIB=np.array([91,85,78,71,64,57,50,43,37])
        ThetaEM=np.array([620,597,569,540,512,484,455,427,404])
        ThetaIM=np.array([155,149,142,135,128,121,114,107,101])
        ThetaEU=np.array([876,853,825,796,768,740,711,683,660])
        ThetaIU=np.array([219,213,206,199,192,185,178,171,165])

        # fig,ax=plt.subplots(figsize=(9,9))
        # ax.scatter(SpikeTimeE[:,0],SpikeTimeE[:,1],s=0.5,c='r',marker='.',linewidths=0,label='SpikeTime')
        # ax.hlines(ThetaE[ThetaEU],0,7000,color='steelblue')
        # ax.hlines(ThetaE[ThetaEM],0,7000,color='darkorange')
        # ax.hlines(ThetaE[ThetaEB],0,7000,color='forestgreen')
        # plt.show()

        EffProfE=ProfE[144:872]
        ReEffProfE=EffProfE.reshape((91,8))
        MaxId=np.where(ReEffProfE==np.max(ReEffProfE))[0][0]*8+4+148
        # print(MaxId)
        # print(ThetaE[MaxId])

        Uid=np.argsort(np.abs(ThetaEU-MaxId))[0]
        Mid=np.argsort(np.abs(ThetaEM-MaxId))[0]
        Bid=np.argsort(np.abs(ThetaEB-MaxId))[0]
        # print(Uid)
        # print(Mid)
        # print(Bid)

        if (np.abs(ThetaEU[Uid]-MaxId) <= np.abs(ThetaEM[Mid]-MaxId)) and (np.abs(ThetaEU[Uid]-MaxId) <= np.abs(ThetaEB[Bid]-MaxId)):
            FinalNeuId=ThetaEU[Uid]
            FinalId=Uid
            VEs[:,TrialID]=SingleUVE[5000:15000,Uid]
            VEIds[TrialID]=ThetaEU[Uid]
            f, Pxx_den = signal.welch(np.mean(SingleUVE[5000:15000,Uid].reshape((400,25)),axis=1), fs=200, window=signal.get_window('hamming',128), noverlap=64, nfft=1024)
            Pxxs[:,TrialID]=Pxx_den
            Mask=(4<=f)*(f<=12)
            peakf[TrialID]=f[np.where(Pxx_den==np.max(Pxx_den))[0][0]]
            peakfintheta[TrialID]=f[np.where(Pxx_den*Mask==np.max(Pxx_den*Mask))[0][0]]
            # print(f[np.where(Pxx_den==np.max(Pxx_den))[0]])
            # print(f[np.where(Pxx_den*Mask==np.max(Pxx_den*Mask))[0]])
        
        if (np.abs(ThetaEB[Bid]-MaxId) <= np.abs(ThetaEM[Mid]-MaxId)) and (np.abs(ThetaEB[Bid]-MaxId) <= np.abs(ThetaEU[Uid]-MaxId)):
            FinalNeuId=ThetaEB[Bid]
            FinalId=Bid
            VEs[:,TrialID]=SingleBVE[5000:15000,Bid]
            VEIds[TrialID]=ThetaEB[Bid]
            f, Pxx_den = signal.welch(np.mean(SingleBVE[5000:15000,Bid].reshape((400,25)),axis=1), fs=200, window=signal.get_window('hamming',128), noverlap=64, nfft=1024)
            Pxxs[:,TrialID]=Pxx_den
            Mask=(4<=f)*(f<=12)
            peakf[TrialID]=f[np.where(Pxx_den==np.max(Pxx_den))[0][0]]
            peakfintheta[TrialID]=f[np.where(Pxx_den*Mask==np.max(Pxx_den*Mask))[0][0]]
            # print(f[np.where(Pxx_den==np.max(Pxx_den))[0]])
            # print(f[np.where(Pxx_den*Mask==np.max(Pxx_den*Mask))[0]])

        if (np.abs(ThetaEM[Mid]-MaxId) <= np.abs(ThetaEU[Uid]-MaxId)) and (np.abs(ThetaEM[Mid]-MaxId) <= np.abs(ThetaEB[Bid]-MaxId)):
            FinalNeuId=ThetaEM[Mid]
            FinalId=Mid
            VEs[:,TrialID]=SingleVE[5000:15000,Mid]
            VEIds[TrialID]=ThetaEM[Mid]
            f, Pxx_den = signal.welch(np.mean(SingleVE[5000:15000,Mid].reshape((400,25)),axis=1), fs=200, window=signal.get_window('hamming',128), noverlap=64, nfft=1024)
            Pxxs[:,TrialID]=Pxx_den
            Mask=(4<=f)*(f<=12)
            peakf[TrialID]=f[np.where(Pxx_den==np.max(Pxx_den))[0][0]]
            peakfintheta[TrialID]=f[np.where(Pxx_den*Mask==np.max(Pxx_den*Mask))[0][0]]
            # print(f[np.where(Pxx_den==np.max(Pxx_den))[0]])
            # print(f[np.where(Pxx_den*Mask==np.max(Pxx_den*Mask))[0]])

        # fig,ax=plt.subplots(figsize=(9,9))
        # ax.plot(np.linspace(0,360,int(NE/8), endpoint=False),np.mean(ProfE.reshape((int(NE/8),8)),axis=1))
        # ax.vlines(ThetaE[ThetaEU],0,12,color='steelblue')
        # ax.vlines(ThetaE[ThetaEM],0,12,color='darkorange')
        # ax.vlines(ThetaE[ThetaEB],0,12,color='forestgreen')
        # ax.vlines(ThetaE[FinalNeuId],0,12,color='firebrick')
        # plt.show()

    # fig,ax=plt.subplots()
    # ax.plot(f[1:-1],10*np.log10(Pxxs[1:-1,8]))
    # plt.show()

    # fig,ax=plt.subplots()
    # ax.plot(np.arange(10000),VEs[:,8])
    # plt.show()

    np.savez_compressed('../Data/DA_Figure3e-g/VEPSD.npz',f=f,Pxxs=Pxxs)
    np.savez_compressed('../Data/DA_Figure3e-g/VEs.npz',VEs=VEs,VEIds=VEIds)

if __name__ == '__main__':
    main()
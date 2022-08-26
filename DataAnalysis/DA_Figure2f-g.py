# -*- coding: utf-8 -*-
"""
@author: Tao WANG, Ziqun WANG
Description: Analyzing data for Figure 2f-g
"""

#### importing packages ####
import numpy as np
from matplotlib import pyplot as plt

def main():
    TauNmda='100'#'75','125','150'
    LoadPath='../Data/DG_Default/sigmaEI30/sigmaIE30/TauNmda'+TauNmda+'/Fext1.8'
    
    NormPVA=np.zeros((7000,100))
    NormPVL=np.zeros((7000,100))
    for TrialID in range(100):
        # print(TrialID)
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
    FinalPVA=np.ones((7000,100))*np.nan
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

    np.save('../Data/DA_Figure2f-g/PVA_TauNmda'+TauNmda+'.npy',FinalPVA)
    # fig,ax=plt.subplots()
    # for i in range(100):
    #     ax.plot(np.arange(7000),FinalPVA[:,i],alpha=0.3)
    # plt.show()

if __name__ == '__main__':
    main()
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 2021

@author: Tao WANG

Description: Analyzing data for Figrue S4a-f
"""

#### importing ####
import numpy as np
from matplotlib import pyplot as plt

def main():
    NE=1024
    ThetaE=2*np.linspace(0,360*(NE-1)/NE,NE)
    ies=['12','18','24','30','36']
    Hie=np.zeros(5)
    Wie=np.zeros(5)
    Sie=np.zeros(5)
    for i in range(3,4):
        AvgProfile=np.zeros(NE)
        for TrialID in range(100):
            Profile=np.load('../Data/DG_Inhibition/sigmaIE'+ies[i]+'/TauNmda100/DevtoCenter0/HalfWidth0/Profile_'+str(TrialID)+'.npz')['arr_0']
            
            v=np.sum(Profile*np.cos(np.pi*ThetaE/180.0))
            h=np.sum(Profile*np.sin(np.pi*ThetaE/180.0))
            Angle=180.0*np.arctan2(h,v)/np.pi
            if h<0: Angle+=360.0
            print(Angle)
            ThetaE-=Angle
            ThetaE=(0<=ThetaE)*(ThetaE<360)*ThetaE+(ThetaE<0)*(ThetaE+360)+(ThetaE>=360)*(ThetaE-360)
            ThetaEID=np.argsort(ThetaE)
            
            # fig=plt.figure()
            # ax=fig.add_subplot(111)
            # ax.plot(ThetaE[ThetaEID]*np.pi/180,Profile[ThetaEID])
            # plt.show()
            AvgProfile+=Profile[ThetaEID]

        AvgProfile/=100

        HalfThetaE=np.linspace(0,180*(NE-1)/NE,NE)

        Hie[i]=np.max(AvgProfile)
        HalfHie=Hie[i]/2
        IncreaseID=np.where((AvgProfile[:-1]<HalfHie)*(AvgProfile[1:]>HalfHie))[0][-1]
        DecreaseID=np.where((AvgProfile[:-1]>HalfHie)*(AvgProfile[1:]<HalfHie))[0][0]
        Wie[i]=180*(DecreaseID+(NE-IncreaseID))/NE
        AvgProfile=np.append(AvgProfile[512:],AvgProfile[:512])
        Sie[i]=((np.sum(AvgProfile*(HalfThetaE-90)**4))/(np.sum(AvgProfile)))/(((np.sum(AvgProfile*(HalfThetaE-90)**2))/(np.sum(AvgProfile)))**2)
        
        fig=plt.figure()
        ax=fig.add_subplot(111)
        ax.plot(AvgProfile)
        plt.show()

    Taus=['75','100','125','150']
    Htau=np.zeros(4)
    Wtau=np.zeros(4)
    Stau=np.zeros(4)
    for i in range(1,2):
        AvgProfile=np.zeros(NE)
        for TrialID in range(100):
            Profile=np.load('../Data/DG_Inhibition/sigmaIE30/TauNmda'+Taus[i]+'/DevtoCenter0/HalfWidth0/Profile_'+str(TrialID)+'.npz')['arr_0']
            
            v=np.sum(Profile*np.cos(np.pi*ThetaE/180.0))
            h=np.sum(Profile*np.sin(np.pi*ThetaE/180.0))
            Angle=180.0*np.arctan2(h,v)/np.pi
            if h<0: Angle+=360.0
            print(Angle)
            ThetaE-=Angle
            ThetaE=(0<=ThetaE)*(ThetaE<360)*ThetaE+(ThetaE<0)*(ThetaE+360)+(ThetaE>=360)*(ThetaE-360)
            ThetaEID=np.argsort(ThetaE)
            
            # fig=plt.figure()
            # ax=fig.add_subplot(111)
            # ax.plot(ThetaE[ThetaEID]*np.pi/180,Profile[ThetaEID])
            # plt.show()
            AvgProfile+=Profile[ThetaEID]

        AvgProfile/=100

        HalfThetaE=np.linspace(0,180*(NE-1)/NE,NE)

        Htau[i]=np.max(AvgProfile)
        HalfHtau=Htau[i]/2
        IncreaseID=np.where((AvgProfile[:-1]<HalfHtau)*(AvgProfile[1:]>HalfHtau))[0][-1]
        DecreaseID=np.where((AvgProfile[:-1]>HalfHtau)*(AvgProfile[1:]<HalfHtau))[0][0]
        Wtau[i]=180*(DecreaseID+(NE-IncreaseID))/NE
        AvgProfile=np.append(AvgProfile[512:],AvgProfile[:512])
        Stau[i]=((np.sum(AvgProfile*(HalfThetaE-90)**4))/(np.sum(AvgProfile)))/(((np.sum(AvgProfile*(HalfThetaE-90)**2))/(np.sum(AvgProfile)))**2)
        
        fig=plt.figure()
        ax=fig.add_subplot(111)
        ax.plot(AvgProfile)
        plt.show()

    np.savez_compressed('../Data/DA_FigureS4a-f/ProfileAbstract.npz',Hie=Hie,Wie=Wie,Sie=Sie,Htau=Htau,Wtau=Wtau,Stau=Stau)

if __name__ == '__main__':
    main()
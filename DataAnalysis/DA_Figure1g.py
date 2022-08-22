# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 2021

@author: Tao WANG

Description: Analyzing data for Figure 1g
"""

#### importing packages ####
import numpy as np
from matplotlib import pyplot as plt

def main():
    SavePath='../Data/DG_Default/sigmaEI30/sigmaIE30/TauNmda100/Fext1.8'
    CurrentIEs=np.zeros((1024,100))
    CurrentBEs=np.zeros((1024,100))
    CurrentLEs=np.zeros((1024,100))
    CurrentEIs=np.zeros((256,100))
    CurrentIIs=np.zeros((256,100))
    CurrentBIs=np.zeros((256,100))
    CurrentLIs=np.zeros((256,100))
    for TrialID in range(100):
        PVs=np.load(SavePath+'/PVs_'+str(TrialID)+'.npz')['arr_0']
        CurrentIE=np.load(SavePath+'/CurrentIE_'+str(TrialID)+'.npz')['arr_0']
        CurrentBE=np.load(SavePath+'/CurrentBE_'+str(TrialID)+'.npz')['arr_0']
        CurrentLE=np.load(SavePath+'/CurrentLE_'+str(TrialID)+'.npz')['arr_0']
        CurrentEI=np.load(SavePath+'/CurrentEI_'+str(TrialID)+'.npz')['arr_0']
        CurrentII=np.load(SavePath+'/CurrentII_'+str(TrialID)+'.npz')['arr_0']
        CurrentBI=np.load(SavePath+'/CurrentBI_'+str(TrialID)+'.npz')['arr_0']
        CurrentLI=np.load(SavePath+'/CurrentLI_'+str(TrialID)+'.npz')['arr_0']

        Idxs=np.where((1500<=PVs[:,0])*(PVs[:,0]<2500))[0]
        print(np.array([PVs[Idxs,2]*np.cos(PVs[Idxs,1]*np.pi/180.0),PVs[Idxs,2]*np.sin(PVs[Idxs,1]*np.pi/180.0)]).shape)
        x,y=np.sum(np.array([PVs[Idxs,2]*np.cos(PVs[Idxs,1]*np.pi/180.0),PVs[Idxs,2]*np.sin(PVs[Idxs,1]*np.pi/180.0)]),axis=1).tolist()
        NormAngle=180.0*np.arctan2(y,x)/np.pi
        if y<0:
            NormAngle+=360.0
        print(NormAngle)

        HalfAngle=NormAngle/2

        NE=1024
        NI=256
        ShiftIdxE=int((90-HalfAngle)*NE/360)
        ShiftIdxI=int((90-HalfAngle)*NI/360)
        if ShiftIdxE>0:
            CurrentIEs[:,TrialID]=np.append(CurrentIE[(NE-ShiftIdxE):],CurrentIE[:(NE-ShiftIdxE)])
            CurrentBEs[:,TrialID]=np.append(CurrentBE[(NE-ShiftIdxE):],CurrentBE[:(NE-ShiftIdxE)])
            CurrentLEs[:,TrialID]=np.append(CurrentLE[(NE-ShiftIdxE):],CurrentLE[:(NE-ShiftIdxE)])
        else:
            CurrentIEs[:,TrialID]=np.append(CurrentIE[ShiftIdxE:],CurrentIE[:ShiftIdxE])
            CurrentBEs[:,TrialID]=np.append(CurrentBE[ShiftIdxE:],CurrentBE[:ShiftIdxE])
            CurrentLEs[:,TrialID]=np.append(CurrentLE[ShiftIdxE:],CurrentLE[:ShiftIdxE])
        
        if ShiftIdxI>0:
            CurrentEIs[:,TrialID]=np.append(CurrentEI[(NI-ShiftIdxI):],CurrentEI[:(NI-ShiftIdxI)])
            CurrentIIs[:,TrialID]=np.append(CurrentII[(NI-ShiftIdxI):],CurrentII[:(NI-ShiftIdxI)])
            CurrentBIs[:,TrialID]=np.append(CurrentBI[(NI-ShiftIdxI):],CurrentBI[:(NI-ShiftIdxI)])
            CurrentLIs[:,TrialID]=np.append(CurrentLI[(NI-ShiftIdxI):],CurrentLI[:(NI-ShiftIdxI)])
        else:
            CurrentEIs[:,TrialID]=np.append(CurrentEI[ShiftIdxI:],CurrentEI[:ShiftIdxI])
            CurrentIIs[:,TrialID]=np.append(CurrentII[ShiftIdxI:],CurrentII[:ShiftIdxI])
            CurrentBIs[:,TrialID]=np.append(CurrentBI[ShiftIdxI:],CurrentBI[:ShiftIdxI])
            CurrentLIs[:,TrialID]=np.append(CurrentLI[ShiftIdxI:],CurrentLI[:ShiftIdxI])

    np.savez_compressed('../Data/DA_Figure1g/Currents.npz',CurrentIEs=CurrentIEs,CurrentBEs=CurrentBEs,CurrentLEs=CurrentLEs,CurrentEIs=CurrentEIs,CurrentIIs=CurrentIIs,CurrentBIs=CurrentBIs,CurrentLIs=CurrentLIs)
    
if __name__ == '__main__':
    main()
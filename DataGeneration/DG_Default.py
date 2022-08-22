# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 2021

@author: Tao WANG

Description: Generating data for Figure 1c-i, 2a-i, 6a-e, S3 in main text.
"""

#### import packages ####
import numpy as np
from scipy import signal
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import sys
import os

#### TrialID ####
global TrialID
TrialID=0
global PVIdx
PVIdx=1

#### simulation settings ####
DeltaT=0.02#ms
T=7000.0#ms

#### network scale ####
NE=1024
NI=256

#### membrane capacitance ####
CME=500.0#pF
CMI=200.0#pF

#### reversal potential for synapses ####
ZE=0.0#mV
ZI=-70.0#mV

#### refractory period ####
tauref_E=2.0#ms
tauref_I=1.0#ms
Maxref_E=int(tauref_E/DeltaT)
Maxref_I=int(tauref_I/DeltaT)

#### leaky conductance ####
GLE=25.0#nS
GLI=20.0#nS
#### reversal potential for leaky channels ####
ZL=-70.0#mV
#### threshold potehtial ####
Eth=-50.0#mV
#### reset potential ####
Eres=-60.0#mV

#### EI connection profile ####
JEIpos=1.5
sigmaEI=float(sys.argv[1])#30.0#degree
#### IE connection profile ####
muIE=90.0
sigmaIE=float(sys.argv[2])#30.0#degree

#### magnesium ion ####
Mg=1.0

#### synaptic conductance ####
GNmdaI=0.584
GGabaE=1.077
GGabaI=2.048

#### time constant for synapses ####
TauAmpa=2.0#ms
TauGaba=10.0#ms
TauX=2.0#ms
TauNmda=float(sys.argv[3])#100.0#ms
AlphaNmda=0.5#kHz

#### background parameters ####
Fext=float(sys.argv[4])#1.8#kHz
GExtE=3.1#nS
GExtI=2.38#nS

#### several values ####
DecX=np.exp(-DeltaT/TauX)
DecA=np.exp(-DeltaT/TauAmpa)
DecG=np.exp(-DeltaT/TauGaba)
DecXh=np.exp(-DeltaT/(2*TauX))
DecAh=np.exp(-DeltaT/(2*TauAmpa))
DecGh=np.exp(-DeltaT/(2*TauGaba))

#### saving settings ####
SavePath='../Data/DG_Default/sigmaEI'+sys.argv[1]+'/sigmaIE'+sys.argv[2]+'/TauNmda'+sys.argv[3]+'/Fext'+sys.argv[4]
isExist=os.path.exists(SavePath)
if not isExist: os.makedirs(SavePath)
np.savetxt(SavePath+'/Parameters.txt',np.array([DeltaT,T,NE,NI,CME,CMI,ZE,ZI,tauref_E,tauref_I,GLE,GLI,ZL,Eth,Eres,JEIpos,sigmaEI,muIE,sigmaIE,Mg,GNmdaI,GGabaE,GGabaI,TauAmpa,TauGaba,TauX,TauNmda,AlphaNmda,Fext,GExtE,GExtI]))

def main():
    #### initializing excitatory cells ####
    VE=np.random.uniform(-60.0,-50.0,NE)
    ThetaE=np.linspace(0,360*(NE-1)/NE,NE)
    RefCountsE=(Maxref_E*np.ones(NE)).astype(np.int32)
    #### initializing interneurons ####
    VI=np.random.uniform(-60.0,-50.0,NI)
    ThetaI=np.linspace(0,360*(NI-1)/NI,NI)
    RefCountsI=(Maxref_I*np.ones(NI)).astype(np.int32)

    #### defining WEI ####
    IntegratePoints=np.linspace(0,180,10000)
    SumExp=np.sum(np.exp(-IntegratePoints**2/(2*sigmaEI**2)))
    JEIneg=(10000-SumExp*JEIpos)/(10000-SumExp)
    DeltaTheta=(ThetaE-ThetaI[0])*np.append(np.ones(int(NE/2)),np.zeros(int(NE/2)))+(360-ThetaE+ThetaI[0])*np.append(np.zeros(int(NE/2)),np.ones(int(NE/2)))
    WEI=JEIneg+(JEIpos-JEIneg)*np.exp(-DeltaTheta**2/(2*sigmaEI**2))
    FWEI=np.fft.fft(WEI)
    #### defining WIE ####
    SumExp=np.sum(np.exp(-(IntegratePoints-muIE)**2/2/sigmaIE**2)*180/10000)
    GIE=180/SumExp
    DeltaTheta0=np.abs(ThetaI[0]-ThetaE[0::4])
    DeltaTheta0=DeltaTheta0*(DeltaTheta0<=180.0)+(360-DeltaTheta0)*(DeltaTheta0>180.0)
    WIE0=GIE*np.exp(-(DeltaTheta0-muIE)**2/2/sigmaIE**2)
    FWIE0=np.fft.fft(WIE0)
    DeltaTheta1=np.abs(ThetaI[0]-ThetaE[1::4])
    DeltaTheta1=DeltaTheta1*(DeltaTheta1<=180.0)+(360-DeltaTheta1)*(DeltaTheta1>180.0)
    WIE1=GIE*np.exp(-(DeltaTheta1-muIE)**2/2/sigmaIE**2)
    FWIE1=np.fft.fft(WIE1)
    DeltaTheta2=np.abs(ThetaI[0]-ThetaE[2::4])
    DeltaTheta2=DeltaTheta2*(DeltaTheta2<=180.0)+(360-DeltaTheta2)*(DeltaTheta2>180.0)
    WIE2=GIE*np.exp(-(DeltaTheta2-muIE)**2/2/sigmaIE**2)
    FWIE2=np.fft.fft(WIE2)
    DeltaTheta3=np.abs(ThetaI[0]-ThetaE[3::4])
    DeltaTheta3=DeltaTheta3*(DeltaTheta3<=180.0)+(360-DeltaTheta3)*(DeltaTheta3>180.0)
    WIE3=GIE*np.exp(-(DeltaTheta3-muIE)**2/2/sigmaIE**2)
    FWIE3=np.fft.fft(WIE3)
    
    #### initializing synaptic state ####
    XNmda=np.zeros(NE)
    SNmda=np.zeros(NE)
    SGaba=np.zeros(NI)
    SExtE=np.zeros(NE)
    SExtI=np.zeros(NI)

    #### recording ####
    SpikeTimeE=np.array([[0,0]])
    SpikeTimeI=np.array([[0,0]])
    VoltageE=np.zeros(NE)
    VoltageI=np.zeros(NI)
    CurrentIE=np.zeros(NE)
    CurrentBE=np.zeros(NE)
    CurrentLE=np.zeros(NE)
    CurrentII=np.zeros(NI)
    CurrentEI=np.zeros(NI)
    CurrentBI=np.zeros(NI)
    CurrentLI=np.zeros(NI)
    Count=0

    #### simulation ####
    STEPs=int(T/DeltaT)
    SingleUVE=np.zeros((int(STEPs/10),9))
    SingleUVI=np.zeros((int(STEPs/10),9))
    SingleVE=np.zeros((int(STEPs/10),9))
    SingleVI=np.zeros((int(STEPs/10),9))
    SingleBVE=np.zeros((int(STEPs/10),9))
    SingleBVI=np.zeros((int(STEPs/10),9))
    for i in range(STEPs):
        if i%10==0:
            SingleBVE[int(i/10),0]=VE[364]
            SingleBVE[int(i/10),1]=VE[341]
            SingleBVE[int(i/10),2]=VE[313]
            SingleBVE[int(i/10),3]=VE[284]
            SingleBVE[int(i/10),4]=VE[256]
            SingleBVE[int(i/10),5]=VE[228]
            SingleBVE[int(i/10),6]=VE[199]
            SingleBVE[int(i/10),7]=VE[171]
            SingleBVE[int(i/10),8]=VE[148]

            SingleBVI[int(i/10),0]=VI[91]
            SingleBVI[int(i/10),1]=VI[85]
            SingleBVI[int(i/10),2]=VI[78]
            SingleBVI[int(i/10),3]=VI[71]
            SingleBVI[int(i/10),4]=VI[64]
            SingleBVI[int(i/10),5]=VI[57]
            SingleBVI[int(i/10),6]=VI[50]
            SingleBVI[int(i/10),7]=VI[43]
            SingleBVI[int(i/10),8]=VI[37]

            SingleVE[int(i/10),0]=VE[620]
            SingleVE[int(i/10),1]=VE[597]
            SingleVE[int(i/10),2]=VE[569]
            SingleVE[int(i/10),3]=VE[540]
            SingleVE[int(i/10),4]=VE[512]
            SingleVE[int(i/10),5]=VE[484]
            SingleVE[int(i/10),6]=VE[455]
            SingleVE[int(i/10),7]=VE[427]
            SingleVE[int(i/10),8]=VE[404]

            SingleVI[int(i/10),0]=VI[155]
            SingleVI[int(i/10),1]=VI[149]
            SingleVI[int(i/10),2]=VI[142]
            SingleVI[int(i/10),3]=VI[135]
            SingleVI[int(i/10),4]=VI[128]
            SingleVI[int(i/10),5]=VI[121]
            SingleVI[int(i/10),6]=VI[114]
            SingleVI[int(i/10),7]=VI[107]
            SingleVI[int(i/10),8]=VI[101]

            SingleUVE[int(i/10),0]=VE[876]
            SingleUVE[int(i/10),1]=VE[853]
            SingleUVE[int(i/10),2]=VE[825]
            SingleUVE[int(i/10),3]=VE[796]
            SingleUVE[int(i/10),4]=VE[768]
            SingleUVE[int(i/10),5]=VE[740]
            SingleUVE[int(i/10),6]=VE[711]
            SingleUVE[int(i/10),7]=VE[683]
            SingleUVE[int(i/10),8]=VE[660]

            SingleUVI[int(i/10),0]=VI[219]
            SingleUVI[int(i/10),1]=VI[213]
            SingleUVI[int(i/10),2]=VI[206]
            SingleUVI[int(i/10),3]=VI[199]
            SingleUVI[int(i/10),4]=VI[192]
            SingleUVI[int(i/10),5]=VI[185]
            SingleUVI[int(i/10),6]=VI[178]
            SingleUVI[int(i/10),7]=VI[171]
            SingleUVI[int(i/10),8]=VI[165]
        
        '''
        if i%100==0:
            print("Please wait... %.2f%% Finished!" % (i*100/STEPs))
        '''
        #### adding stimulus ####
        Istim=0.0
        
        ############ first step of 2nd-RK ############
        ######## excitatory cells ########
        #### in ref period? ####
        OutrefE=(RefCountsE==Maxref_E)
        #### calculating Gaba IE currents ####
        FSGaba=np.fft.fft(SGaba)
        FWIE0_SGaba=FWIE0*FSGaba
        SumGIE0=np.real(np.fft.ifft(FWIE0_SGaba))
        FWIE1_SGaba=FWIE1*FSGaba
        SumGIE1=np.real(np.fft.ifft(FWIE1_SGaba))
        FWIE2_SGaba=FWIE2*FSGaba
        SumGIE2=np.real(np.fft.ifft(FWIE2_SGaba))
        FWIE3_SGaba=FWIE3*FSGaba
        SumGIE3=np.real(np.fft.ifft(FWIE3_SGaba))
        SumGIE=np.zeros(NE)
        SumGIE[0::4]=SumGIE0
        SumGIE[1::4]=SumGIE1
        SumGIE[2::4]=SumGIE2
        SumGIE[3::4]=SumGIE3
        IGabaE=(VE-ZI)*GGabaE*SumGIE
        #### calculating background currents ####
        IextE=(VE-ZE)*GExtE*SExtE
        #### calculating leaky currents ####
        IleakE=GLE*(VE-ZL)
        #### integration ####
        K1E=(Istim-IGabaE-IextE-IleakE)/CME
        VEtemp=VE+DeltaT*K1E

        ######## interneuron ########
        #### in ref period? ####
        OutrefI=(RefCountsI==Maxref_I)
        #### calculating Nmda EI currents ####
        FSNmda=np.fft.fft(SNmda)
        FWEI_SNmda=FWEI*FSNmda
        SumNEI=np.real(np.fft.ifft(FWEI_SNmda))[0::4]
        INmdaI=(VI-ZE)*GNmdaI*SumNEI/(1.0+Mg*np.exp(-0.062*VI)/3.57)
        #### calculating Gaba II currents ####
        SumGII=np.sum(SGaba)
        IGabaI=(VI-ZI)*GGabaI*SumGII
        #### calculating background currents ####
        IextI=(VI-ZE)*GExtI*SExtI
        #### calculating leaky currents ####
        IleakI=GLI*(VI-ZL)
        #### integration ####
        K1I=(-INmdaI-IGabaI-IextI-IleakI)/CMI
        VItemp=VI+DeltaT*K1I

        ######## synaptic states ########
        K1S=-SNmda/TauNmda+AlphaNmda*XNmda*(1.0-SNmda)
        SNmdaTemp=SNmda+DeltaT*K1S
        XNmdaTemp=XNmda*DecX
        SExtETemp=SExtE*DecA

        SGabaTemp=SGaba*DecG
        SExtITemp=SExtI*DecA

        ############ second step of 2nd-RK ############
        ######## excitatory cells ########
        #### calculating Gaba IE currents ####
        FSGabaTemp=np.fft.fft(SGabaTemp)
        FWIE0_SGabaTemp=FWIE0*FSGabaTemp
        SumGIE0Temp=np.real(np.fft.ifft(FWIE0_SGabaTemp))
        FWIE1_SGabaTemp=FWIE1*FSGabaTemp
        SumGIE1Temp=np.real(np.fft.ifft(FWIE1_SGabaTemp))
        FWIE2_SGabaTemp=FWIE2*FSGabaTemp
        SumGIE2Temp=np.real(np.fft.ifft(FWIE2_SGabaTemp))
        FWIE3_SGabaTemp=FWIE3*FSGabaTemp
        SumGIE3Temp=np.real(np.fft.ifft(FWIE3_SGabaTemp))
        SumGIETemp=np.zeros(NE)
        SumGIETemp[0::4]=SumGIE0Temp
        SumGIETemp[1::4]=SumGIE1Temp
        SumGIETemp[2::4]=SumGIE2Temp
        SumGIETemp[3::4]=SumGIE3Temp
        IGabaE=(VEtemp-ZI)*GGabaE*SumGIETemp
        #### calculating background currents ####
        IextE=(VEtemp-ZE)*GExtE*SExtETemp
        #### calculating leaky currents ####
        IleakE=GLE*(VEtemp-ZL)
        #### integration ####
        K2E=(Istim-IGabaE-IextE-IleakE)/CME
        VE=VE+(DeltaT/2)*(K1E+K2E)*OutrefE

        ######## interneurons ########
        #### calculating Nmda EI currents ####
        FSNmdaTemp=np.fft.fft(SNmdaTemp)
        FWEI_SNmdaTemp=FWEI*FSNmdaTemp
        SumNEITemp=np.real(np.fft.ifft(FWEI_SNmdaTemp))[0::4]
        INmdaI=(VItemp-ZE)*GNmdaI*SumNEITemp/(1.0+Mg*np.exp(-0.062*VItemp)/3.57)
        #### calculating Gaba II currents ####
        SumGIITemp=np.sum(SGabaTemp)
        # SumGIITemp=SumGII*DecGh
        IGabaI=(VItemp-ZI)*GGabaI*SumGIITemp
        #### calculating background currents ####
        IextI=(VItemp-ZE)*GExtI*SExtITemp
        #### calculating leaky currents ####
        IleakI=GLI*(VItemp-ZL)
        #### integration ####
        K2I=(-INmdaI-IGabaI-IextI-IleakI)/CMI
        VI=VI+(DeltaT/2)*(K1I+K2I)*OutrefI

        ######## synaptic states ########
        K2S=(-SNmdaTemp/TauNmda+AlphaNmda*XNmdaTemp*(1.0-SNmdaTemp))
        SNmda=SNmda+(DeltaT/2)*(K1S+K2S)
        XNmda=XNmda*DecX
        SExtE=SExtE*DecA

        SGaba=SGaba*DecG
        SExtI=SExtI*DecA
        
        ############ spike? ############
        ######## excitatory cells ########
        FlagE=(VE>=Eth)
        VE=FlagE*Eres+(~FlagE)*VE
        RefCountsE=0*FlagE+(~FlagE)*OutrefE*RefCountsE+(~FlagE)*(~OutrefE)*(RefCountsE+1)
        XNmda+=FlagE#*DecX

        ######## interneurons ########
        FlagI=(VI>=Eth)
        VI=FlagI*Eres+(~FlagI)*VI
        RefCountsI=0*FlagI+(~FlagI)*OutrefI*RefCountsI+(~FlagI)*(~OutrefI)*(RefCountsI+1)
        SGaba+=FlagI#*DecG

        ######## background spike? ########
        FlagB2E=np.random.poisson((Fext*DeltaT),NE)
        FlagB2I=np.random.poisson((Fext*DeltaT),NI)
        SExtE+=FlagB2E
        SExtI+=FlagB2I

        ######## record ########
        for Eid in np.where(FlagE)[0]:
            SpikeTimeE=np.append(SpikeTimeE,np.array([[(i*DeltaT),ThetaE[Eid]]]),axis=0)
        for Iid in np.where(FlagI)[0]:
            SpikeTimeI=np.append(SpikeTimeI,np.array([[(i*DeltaT),ThetaI[Iid]]]),axis=0)
        if 1500<(i*DeltaT) and (i*DeltaT)<2500.0:
            CurrentIE+=IGabaE
            CurrentBE+=IextE
            CurrentLE+=IleakE
            CurrentII+=IGabaI
            CurrentEI+=INmdaI
            CurrentBI+=IextI
            CurrentLI+=IleakI
            Count+=1

        ############ end of this step ############

    #### analysis ####
    #### calculating the firing profile and population vector ####
    ProfE=np.zeros(NE)
    PVs=np.array([[0,180,0]])
    TW=np.array([[0,180]])
    v=0
    h=0
    for (t,Theta) in SpikeTimeE:
        if 1500<t and t<2500:
            ProfE[int(Theta*NE/360)]+=1
        TW=np.append(TW,[[t,Theta]],axis=0)
        v+=np.cos(np.pi*2*Theta/180.0)
        h+=np.sin(np.pi*2*Theta/180.0)
        while TW[0,0]<(t-400.0):
            v-=np.cos(np.pi*2*TW[0,1]/180.0)
            h-=np.sin(np.pi*2*TW[0,1]/180.0)
            TW=np.delete(TW,0,0)
        PVA=180.0*np.arctan2(h,v)/np.pi
        PVL=np.sqrt(v**2+h**2)/0.4
        if h<0:
            PVA+=360.0
        PVs=np.append(PVs,[[t,PVA,PVL]],axis=0)
    ProfE/=1

    #### save initial recordings ####
    np.savez_compressed(SavePath+'/PVs_'+str(TrialID)+'.npz',PVs)
    np.savez_compressed(SavePath+'/Profile_'+str(TrialID)+'.npz',ProfE)
    np.savez_compressed(SavePath+'/SpikeTimeE_'+str(TrialID)+'.npz',SpikeTimeE)
    np.savez_compressed(SavePath+'/SpikeTimeI_'+str(TrialID)+'.npz',SpikeTimeI)
    np.savez_compressed(SavePath+'/CurrentIE_'+str(TrialID)+'.npz',CurrentIE/Count)
    np.savez_compressed(SavePath+'/CurrentBE_'+str(TrialID)+'.npz',CurrentBE/Count)
    np.savez_compressed(SavePath+'/CurrentLE_'+str(TrialID)+'.npz',CurrentLE/Count)
    np.savez_compressed(SavePath+'/CurrentEI_'+str(TrialID)+'.npz',CurrentEI/Count)
    np.savez_compressed(SavePath+'/CurrentII_'+str(TrialID)+'.npz',CurrentII/Count)
    np.savez_compressed(SavePath+'/CurrentBI_'+str(TrialID)+'.npz',CurrentBI/Count)
    np.savez_compressed(SavePath+'/CurrentLI_'+str(TrialID)+'.npz',CurrentLI/Count)
    np.savez_compressed(SavePath+'/SingleBVE_'+str(TrialID)+'.npz',SingleBVE)
    np.savez_compressed(SavePath+'/SingleBVI_'+str(TrialID)+'.npz',SingleBVI)
    np.savez_compressed(SavePath+'/SingleVE_'+str(TrialID)+'.npz',SingleVE)
    np.savez_compressed(SavePath+'/SingleVI_'+str(TrialID)+'.npz',SingleVI)
    np.savez_compressed(SavePath+'/SingleUVE_'+str(TrialID)+'.npz',SingleUVE)
    np.savez_compressed(SavePath+'/SingleUVI_'+str(TrialID)+'.npz',SingleUVI)

    #### PSD analysis for spike count ####
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

    np.savez_compressed(SavePath+'/PSD_'+str(TrialID)+'.npz',Freq=f,PE=Pxx_den)

if __name__=='__main__':
    for trial in range(100):
        main()
        TrialID+=1
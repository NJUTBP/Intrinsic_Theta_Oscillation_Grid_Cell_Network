# -*- coding: utf-8 -*-
"""
@author: Tao WANG, Ziqun WANG
Description: Generating data for Figure S1.
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
PVIdx=2

#### simulation settings ####
dt=0.02#ms
T=8000.0#ms

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
Maxref_E=int(tauref_E/dt)
Maxref_I=int(tauref_I/dt)
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
JEIpos=1.6
sigmaEI=15.0
#### IE connection profile ####
muIE=60.0
sigmaIE=15.0
#### magnesium ion ####
Mg=1.0
#### synaptic conductance ####
GNmdaI=0.4
GGabaE=2.4
GGabaI=0.04
#### time constant for synapses ####
TauAmpa=2.0#ms
TauGaba=10.0#ms
TauX=2.0#ms
TauNmda=100.0#ms
AlphaNmda=0.5#kHz

#### background parameters ####
IB2E=-750.0#pA
IB2I=-225.0#pA
IMS2I=100.0#pA

#### noise parameters ####
TauNoise=2.0#ms
SigmaNoise=150.0#pA

#### Stimulus input ####
Stim2EStart=0
Stim2EEnd=500
IS2E0=np.zeros(NE)
IS2E0[int(NE/6)-int(NE/24):int(NE/6)+int(NE/24)]=200#pA
IS2E0[int(NE*3/6)-int(NE/24):int(NE*3/6)+int(NE/24)]=200#pA
IS2E0[int(NE*5/6)-int(NE/24):int(NE*5/6)+int(NE/24)]=200#pA

#### LFP sampling rate ####
SamplingRate=200
TbinNum=int(T*SamplingRate/1000)

#### several values ####
DecX=np.exp(-dt/TauX)
DecA=np.exp(-dt/TauAmpa)
DecG=np.exp(-dt/TauGaba)
DecXh=np.exp(-dt/(2*TauX))
DecAh=np.exp(-dt/(2*TauAmpa))
DecGh=np.exp(-dt/(2*TauGaba))

#### recording settings ####
Parameters={'dt':dt,'T':T,
            'NE':NE,'NI':NI,
            'CME':CME,'CMI':CMI,'ZE':ZE,'ZI':ZI,
            'tauref_E':tauref_E,'tauref_I':tauref_I,
            'GLE':GLE,'GLI':GLI,'ZL':ZL,'Eth':Eth,'Eres':Eres,
            'JEIpos':JEIpos,'sigmaEI':sigmaEI,'muIE':muIE,'sigmaIE':sigmaIE,
            'Mg':Mg,
            'GNmdaI':GNmdaI,'GGabaE':GGabaE,'GGabaI':GGabaI,
            'TauAmpa':TauAmpa,'TauGaba':TauGaba,'TauX':TauX,'TauNmda':TauNmda,'AlphaNmda':AlphaNmda,
            'IB2E':IB2E,'IB2I':IB2I,'IMS2I':IMS2I,
            'TauNoise':TauNoise,'SigmaNoise':SigmaNoise,
            'SamplingRate':SamplingRate,
            'Stim2EStart':Stim2EStart,'Stim2EEnd':Stim2EEnd,'IS2E0':IS2E0}

FilePath='../Data/ThreeBumps/'
isExist=os.path.exists(FilePath)
if not isExist: os.makedirs(FilePath)

def main(TrialID):
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

    #### noise currents ####
    IN2E=np.zeros(NE)
    IN2I=np.zeros(NI)

    #### recording ####
    #### spike time of E and I neuron
    SpikeTimeE=np.array([[0,0]])
    SpikeTimeI=np.array([[0,0]])
    #### voltage and currents of E and I neuron during given time period
    VoltageE=np.zeros(NE)
    VoltageI=np.zeros(NI)
    CurrentEE=np.zeros(NE)
    CurrentIE=np.zeros(NE)
    CurrentBE=IB2E*np.ones(NE)
    CurrentNE=np.zeros(NE)
    CurrentLE=np.zeros(NE)
    CurrentII=np.zeros(NI)
    CurrentEI=np.zeros(NI)
    CurrentBI=IB2I*np.ones(NI)
    CurrentMS=IMS2I*np.ones(NI)
    CurrentNI=np.zeros(NI)
    CurrentLI=np.zeros(NI)
    CurrentCount=0
    #### GABA currents received by E neurons versus time
    Gaba2E=np.zeros((NE,TbinNum))
    Gaba2E_RecordInterval=int(1000/SamplingRate/dt)
    #### noise currents received by E neurons versus time
    # Nois2E=np.zeros((int(NE/8),TbinNum))
    # Nois2I=np.zeros((int(NI/8),TbinNum))
    #### membrane potentials of selected E and I neurons
    VmE=np.zeros((int(NE/8),TbinNum))
    VmI=np.zeros((int(NI/8),TbinNum))
    VmE_RecNeu_ID=np.arange(0,NE,8)
    VmI_RecNeu_ID=np.arange(0,NI,8)

    #### simulation ####
    STEPs=int(T/dt)
    for i in range(STEPs):
        # if i%(int(5*T))==0:
        #     print("Please wait... %.2f%% Finished!" % (i*100/STEPs))
        
        #### adding stimulus ####
        IS2E=np.zeros(NE)
        if Stim2EStart<(i*dt) and (i*dt)<Stim2EEnd:
            IS2E=IS2E0
        
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
        #### calculating leaky currents ####
        ILeakE=GLE*(VE-ZL)
        #### integration ####
        K1E=(IS2E-IGabaE-IB2E-IN2E-ILeakE)/CME
        VETemp=VE+dt*K1E

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
        #### calculating leaky currents ####
        ILeakI=GLI*(VI-ZL)
        #### integration ####
        K1I=(-INmdaI-IGabaI-IB2I-IMS2I-IN2I-ILeakI)/CMI
        VITemp=VI+dt*K1I

        ######## synaptic states ########
        K1S=-SNmda/TauNmda+AlphaNmda*XNmda*(1.0-SNmda)
        SNmdaTemp=SNmda+dt*K1S
        XNmdaTemp=XNmda*DecX

        SGabaTemp=SGaba*DecG

        IN2ETemp=IN2E-dt*(IN2E-0)/TauNoise
        IN2ITemp=IN2I-dt*(IN2I-0)/TauNoise

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
        IGabaE=(VETemp-ZI)*GGabaE*SumGIETemp
        #### calculating leaky currents ####
        ILeakE=GLE*(VETemp-ZL)
        #### integration ####
        K2E=(IS2E-IGabaE-IB2E-IN2ETemp-ILeakE)/CME
        VE=VE+(dt/2)*(K1E+K2E)*OutrefE

        ######## interneurons ########
        #### calculating Nmda EI currents ####
        FSNmdaTemp=np.fft.fft(SNmdaTemp)
        FWEI_SNmdaTemp=FWEI*FSNmdaTemp
        SumNEITemp=np.real(np.fft.ifft(FWEI_SNmdaTemp))[0::4]
        INmdaI=(VITemp-ZE)*GNmdaI*SumNEITemp/(1.0+Mg*np.exp(-0.062*VITemp)/3.57)
        #### calculating Gaba II currents ####
        SumGIITemp=np.sum(SGabaTemp)
        IGabaI=(VITemp-ZI)*GGabaI*SumGIITemp
        #### calculating leaky currents ####
        ILeakI=GLI*(VITemp-ZL)
        #### integration ####
        K2I=(-INmdaI-IGabaI-IB2I-IMS2I-IN2ITemp-ILeakI)/CMI
        VI=VI+(dt/2)*(K1I+K2I)*OutrefI

        ######## synaptic states ########
        K2S=(-SNmdaTemp/TauNmda+AlphaNmda*XNmdaTemp*(1.0-SNmdaTemp))
        SNmda=SNmda+(dt/2)*(K1S+K2S)
        XNmda=XNmda*DecX

        SGaba=SGaba*DecG

        IN2E=IN2E-(dt/2)*(IN2E+IN2ETemp-2*0)/TauNoise+np.random.normal(size=NE)*SigmaNoise*np.sqrt(dt)/np.sqrt(TauNoise)
        IN2I=IN2I-(dt/2)*(IN2I+IN2ITemp-2*0)/TauNoise+np.random.normal(size=NI)*SigmaNoise*np.sqrt(dt)/np.sqrt(TauNoise)
        
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

        ######## record ########
        for Eid in np.where(FlagE)[0]:
            SpikeTimeE=np.append(SpikeTimeE,np.array([[(i*dt),ThetaE[Eid]]]),axis=0)
        for Iid in np.where(FlagI)[0]:
            SpikeTimeI=np.append(SpikeTimeI,np.array([[(i*dt),ThetaI[Iid]]]),axis=0)
        if (Stim2EEnd+1000)<(i*dt) and (i*dt)<(Stim2EEnd+2000):
            CurrentIE+=IGabaE
            CurrentNE+=IN2E
            CurrentLE+=ILeakE
            CurrentII+=IGabaI
            CurrentEI+=INmdaI
            CurrentNI+=IN2I
            CurrentLI+=ILeakI
            CurrentCount+=1
        if i%(Gaba2E_RecordInterval)==0:
            Gaba2E[:,int(i/Gaba2E_RecordInterval)]=IGabaE
            VmE[:,int(i/Gaba2E_RecordInterval)]=VE[VmE_RecNeu_ID]
            VmI[:,int(i/Gaba2E_RecordInterval)]=VI[VmI_RecNeu_ID]
            # Nois2E[:,int(i/Gaba2E_RecordInterval)]=IN2E[VmE_RecNeu_ID]
            # Nois2I[:,int(i/Gaba2E_RecordInterval)]=IN2I[VmI_RecNeu_ID]

        ############ end of this step ############

    #### analysis ####
    #### calculating the firing profile ####
    ProfE=np.zeros(NE)
    ProfI=np.zeros(NI)
    ProfE,_=np.histogram((SpikeTimeE[((Stim2EEnd+1000)<SpikeTimeE[:,0])*(SpikeTimeE[:,0]<(Stim2EEnd+2000)),1]*NE/360),bins=np.linspace(-0.5,NE-0.5,NE+1))
    ProfI,_=np.histogram((SpikeTimeI[((Stim2EEnd+1000)<SpikeTimeI[:,0])*(SpikeTimeI[:,0]<(Stim2EEnd+2000)),1]*NI/360),bins=np.linspace(-0.5,NI-0.5,NI+1))
    ProfE=ProfE/1
    ProfI=ProfI/1

    #### calculating the population activity vector ####
    PVs=np.array([[0,180,0]])
    TW=np.array([[0,180]])
    NormPVs=np.zeros((TbinNum,2))
    Current_PV_bin=0
    v=0
    h=0
    for (t,Theta) in SpikeTimeE:
        if t>(Current_PV_bin*1000/SamplingRate):
            NormPVs[Current_PV_bin,:]=np.array([PVA,PVL])
            Current_PV_bin+=1
        TW=np.append(TW,[[t,Theta]],axis=0)
        v+=np.cos(np.pi*3*Theta/180.0)
        h+=np.sin(np.pi*3*Theta/180.0)
        # while TW[0,0]<((Current_PV_bin*1000/SamplingRate)-400.0):
        while TW[0,0]<(t-400.0):
            v-=np.cos(np.pi*3*TW[0,1]/180.0)
            h-=np.sin(np.pi*3*TW[0,1]/180.0)
            TW=np.delete(TW,0,0)
        PVA=180.0*np.arctan2(h,v)/np.pi
        PVL=np.sqrt(v**2+h**2)/0.4/NE
        if h<0:
            PVA+=360.0
        PVs=np.append(PVs,[[t,PVA,PVL]],axis=0)
    
    Tbinedges=np.linspace(0,T,TbinNum+1,endpoint=True)
    Tbins=(Tbinedges[1:]+Tbinedges[:-1])/2

    #### calculating the LFP proxy and its PSD ####
    SumGaba2E=np.mean(Gaba2E,axis=0)
    Freq, PSD_SGabaE = signal.welch(SumGaba2E[(1*SamplingRate):], fs=SamplingRate, window=signal.get_window('hamming',256), noverlap=128, nfft=1024)
    
    #### calculating the spike histogram of all neurons and its PSD ####
    ESpikeHist,_=np.histogram(SpikeTimeE[:,0],bins=Tbinedges)
    ISpikeHist,_=np.histogram(SpikeTimeI[:,0],bins=Tbinedges)
    _, PSD_ESpike = signal.welch(ESpikeHist[(1*SamplingRate):], fs=SamplingRate, window=signal.get_window('hamming',256), noverlap=128, nfft=1024)
    _, PSD_ISpike = signal.welch(ISpikeHist[(1*SamplingRate):], fs=SamplingRate, window=signal.get_window('hamming',256), noverlap=128, nfft=1024)
    
    #### calculating the PSDs of membrane potentials of all neurons ####
    _, PSD_VmE = signal.welch(VmE[:,(1*SamplingRate):], fs=SamplingRate, window=signal.get_window('hamming',256), noverlap=128, nfft=1024, axis=-1)
    _, PSD_VmI = signal.welch(VmI[:,(1*SamplingRate):], fs=SamplingRate, window=signal.get_window('hamming',256), noverlap=128, nfft=1024, axis=-1)
    
    #### calculating PSD power ####
    #### defining theta and delta range
    Freq_theta=(4<Freq)*(Freq<=12)
    Freq_delta=(2<Freq)*(Freq<=4)

    #### calculating the major peak within 0~100 Hz
    MaxPeakID_PSD_SGabaE=np.argmax(PSD_SGabaE[Freq<50])
    MaxPeakID_PSD_ESpike=np.argmax(PSD_ESpike[Freq<50])
    MaxPeakID_PSD_ISpike=np.argmax(PSD_ISpike[Freq<50])
    
    #### calculating theta power relative to the 0~50 Hz and to delta range
    thetaPower_SGabaE=np.sum(PSD_SGabaE*Freq_theta)
    thetaPower_ESpike=np.sum(PSD_ESpike*Freq_theta)
    thetaPower_ISpike=np.sum(PSD_ISpike*Freq_theta)

    Relative_thetaPower_SGabaE=np.mean(PSD_SGabaE[Freq_theta])/np.mean(PSD_SGabaE[Freq<50])
    Relative_thetaPower_ESpike=np.mean(PSD_ESpike[Freq_theta])/np.mean(PSD_ESpike[Freq<50])
    Relative_thetaPower_ISpike=np.mean(PSD_ISpike[Freq_theta])/np.mean(PSD_ISpike[Freq<50])

    Fraction_thetaPower_SGabaE=np.sum(PSD_SGabaE[Freq_theta])/np.sum(PSD_SGabaE[Freq<50])
    Fraction_thetaPower_ESpike=np.sum(PSD_ESpike[Freq_theta])/np.sum(PSD_ESpike[Freq<50])
    Fraction_thetaPower_ISpike=np.sum(PSD_ISpike[Freq_theta])/np.sum(PSD_ISpike[Freq<50])

    deltaPower_SGabaE=np.sum(PSD_SGabaE*Freq_delta)
    deltaPower_ESpike=np.sum(PSD_ESpike*Freq_delta)
    deltaPower_ISpike=np.sum(PSD_ISpike*Freq_delta)

    theta2delta_SGabaE=thetaPower_SGabaE/deltaPower_SGabaE
    theta2delta_ESpike=thetaPower_ESpike/deltaPower_ESpike
    theta2delta_ISpike=thetaPower_ISpike/deltaPower_ISpike
    
    #### calculating the theta peak and the peak strength 
    MaxThetaPeakID_PSD_SGabaE=np.argmax(PSD_SGabaE*Freq_theta)
    MaxThetaPeakID_PSD_ESpike=np.argmax(PSD_ESpike*Freq_theta)
    MaxThetaPeakID_PSD_ISpike=np.argmax(PSD_ISpike*Freq_theta)

    FreqAroundThetaPeak_SGabaE=((Freq[MaxThetaPeakID_PSD_SGabaE]-1)<Freq)*(Freq<=(Freq[MaxThetaPeakID_PSD_SGabaE]+1))
    FreqAroundThetaPeak_ESpike=((Freq[MaxThetaPeakID_PSD_ESpike]-1)<Freq)*(Freq<=(Freq[MaxThetaPeakID_PSD_ESpike]+1))
    FreqAroundThetaPeak_ISpike=((Freq[MaxThetaPeakID_PSD_ISpike]-1)<Freq)*(Freq<=(Freq[MaxThetaPeakID_PSD_ISpike]+1))

    thetaPeakPower_SGabaE=np.sum(PSD_SGabaE[FreqAroundThetaPeak_SGabaE])
    thetaPeakPower_ESpike=np.sum(PSD_ESpike[FreqAroundThetaPeak_ESpike])
    thetaPeakPower_ISpike=np.sum(PSD_ISpike[FreqAroundThetaPeak_ISpike])

    Relative_thetaPeakPower_SGabaE=np.mean(PSD_SGabaE[FreqAroundThetaPeak_SGabaE])/np.mean(PSD_SGabaE[Freq<50])
    Relative_thetaPeakPower_ESpike=np.mean(PSD_ESpike[FreqAroundThetaPeak_ESpike])/np.mean(PSD_ESpike[Freq<50])
    Relative_thetaPeakPower_ISpike=np.mean(PSD_ISpike[FreqAroundThetaPeak_ISpike])/np.mean(PSD_ISpike[Freq<50])
    
    Fraction_thetaPeakPower_SGabaE=np.sum(PSD_SGabaE[FreqAroundThetaPeak_SGabaE])/np.sum(PSD_SGabaE[Freq<50])
    Fraction_thetaPeakPower_ESpike=np.sum(PSD_ESpike[FreqAroundThetaPeak_ESpike])/np.sum(PSD_ESpike[Freq<50])
    Fraction_thetaPeakPower_ISpike=np.sum(PSD_ISpike[FreqAroundThetaPeak_ISpike])/np.sum(PSD_ISpike[Freq<50])

    MaxPeakID_PSD_VmE=np.argmax(PSD_VmE[:,Freq<50],axis=1)
    thetaPower_VmE=np.sum(PSD_VmE[:,Freq_theta],axis=1)
    Relative_thetaPower_VmE=np.mean(PSD_VmE[:,Freq_theta],axis=1)/np.mean(PSD_VmE[:,Freq<50],axis=1)
    Fraction_thetaPower_VmE=np.sum(PSD_VmE[:,Freq_theta],axis=1)/np.sum(PSD_VmE[:,Freq<50],axis=1)
    deltaPower_VmE=np.sum(PSD_VmE[:,Freq_delta],axis=1)
    theta2delta_VmE=thetaPower_VmE/deltaPower_VmE
    MaxThetaPeakID_PSD_VmE=np.zeros(int(NE/8)).astype(np.int32)
    thetaPeakPower_VmE=np.zeros(int(NE/8))
    Relative_thetaPeakPower_VmE=np.zeros(int(NE/8))
    Fraction_thetaPeakPower_VmE=np.zeros(int(NE/8))
    for i in range(int(NE/8)):
        MaxThetaPeakID_PSD_VmE[i]=np.argmax(PSD_VmE[i,:]*Freq_theta)
        FreqAroundThetaPeak_VmE=((Freq[MaxThetaPeakID_PSD_VmE[i]]-1)<Freq)*(Freq<=(Freq[MaxThetaPeakID_PSD_VmE[i]]+1))
        thetaPeakPower_VmE[i]=np.sum(PSD_VmE[i,FreqAroundThetaPeak_VmE])
        Relative_thetaPeakPower_VmE[i]=np.mean(PSD_VmE[i,FreqAroundThetaPeak_VmE])/np.mean(PSD_VmE[i,Freq<50])
        Fraction_thetaPeakPower_VmE[i]=np.sum(PSD_VmE[i,FreqAroundThetaPeak_VmE])/np.sum(PSD_VmE[i,Freq<50])

    MaxPeakID_PSD_VmI=np.argmax(PSD_VmI[:,Freq<50],axis=1)
    thetaPower_VmI=np.sum(PSD_VmI[:,Freq_theta],axis=1)
    Relative_thetaPower_VmI=np.mean(PSD_VmI[:,Freq_theta],axis=1)/np.mean(PSD_VmI[:,Freq<50],axis=1)
    Fraction_thetaPower_VmI=np.sum(PSD_VmI[:,Freq_theta],axis=1)/np.sum(PSD_VmI[:,Freq<50],axis=1)
    deltaPower_VmI=np.sum(PSD_VmI[:,Freq_delta],axis=1)
    theta2delta_VmI=thetaPower_VmI/deltaPower_VmI
    MaxThetaPeakID_PSD_VmI=np.zeros(int(NI/8)).astype(np.int32)
    thetaPeakPower_VmI=np.zeros(int(NI/8))
    Relative_thetaPeakPower_VmI=np.zeros(int(NI/8))
    Fraction_thetaPeakPower_VmI=np.zeros(int(NI/8))
    for i in range(int(NI/8)):
        MaxThetaPeakID_PSD_VmI[i]=np.argmax(PSD_VmI[i,:]*Freq_theta)
        FreqAroundThetaPeak_VmI=((Freq[MaxThetaPeakID_PSD_VmI[i]]-1)<Freq)*(Freq<=(Freq[MaxThetaPeakID_PSD_VmI[i]]+1))
        thetaPeakPower_VmI[i]=np.sum(PSD_VmI[i,FreqAroundThetaPeak_VmI])
        Relative_thetaPeakPower_VmI[i]=np.mean(PSD_VmI[i,FreqAroundThetaPeak_VmI])/np.mean(PSD_VmI[i,Freq<50])
        Fraction_thetaPeakPower_VmI[i]=np.sum(PSD_VmI[i,FreqAroundThetaPeak_VmI])/np.sum(PSD_VmI[i,Freq<50])

    #### calculating spike autocorrelation for each neuron ####
    ACEs=np.zeros((NE,1*SamplingRate))
    ACIs=np.zeros((NI,1*SamplingRate))
    ACbinedges=np.linspace(0,1,SamplingRate*1+1)
    ACbins=(ACbinedges[:-1]+ACbinedges[1:])/2
    EHist2D,_,_=np.histogram2d(SpikeTimeE[:,0],SpikeTimeE[:,1],bins=[np.linspace(0,T,TbinNum+1),np.linspace(-180/NE,360-180/NE,NE+1)])
    IHist2D,_,_=np.histogram2d(SpikeTimeI[:,0],SpikeTimeI[:,1],bins=[np.linspace(0,T,TbinNum+1),np.linspace(-180/NI,360-180/NI,NI+1)])
    ACEs[:,0]=np.sum(EHist2D*EHist2D,axis=0)
    ACIs[:,0]=np.sum(IHist2D*IHist2D,axis=0)
    for delay in range(1,SamplingRate):
        ACEs[:,delay]=np.sum(EHist2D[delay:]*EHist2D[:-delay],axis=0)
        ACIs[:,delay]=np.sum(IHist2D[delay:]*IHist2D[:-delay],axis=0)

    #### calculating instantaneous phase of LFP proxy and spike histogram ####
    sos_theta=signal.butter(10, [4,12], 'bandpass', fs=SamplingRate, output='sos')
    filt_SGabaE=signal.sosfiltfilt(sos_theta,SumGaba2E)
    hilbert_SGabaE=signal.hilbert(filt_SGabaE)
    Phase_SGabaE=np.angle(hilbert_SGabaE)

    filt_ESpikeHist=signal.sosfiltfilt(sos_theta,ESpikeHist)
    hilbert_ESpikeHist=signal.hilbert(filt_ESpikeHist)
    Phase_ESpikeHist=np.angle(hilbert_ESpikeHist)

    filt_ISpikeHist=signal.sosfiltfilt(sos_theta,ISpikeHist)
    hilbert_ISpikeHist=signal.hilbert(filt_ISpikeHist)
    Phase_ISpikeHist=np.angle(hilbert_ISpikeHist)
    
    #### calculating spike phase relative to the LFP proxy theta and phase locking index ####
    SpikeTimePhaseE=np.zeros((SpikeTimeE.shape[0],3))
    SpikeTimePhaseE[:,0:2]=SpikeTimeE
    PhaseLockingIndexE=np.zeros((NE,2))
    SumSinE=np.zeros(NE)
    SumCosE=np.zeros(NE)
    CountsE=np.zeros(NE)
    for k in range(SpikeTimeE.shape[0]):
        TbinID=np.argmin(np.abs(SpikeTimeE[k,0]-TauGaba-Tbins))
        SpikeTimePhaseE[k,2]=Phase_SGabaE[TbinID]

        if SpikeTimeE[k,0]>(Stim2EEnd+500):
            NeuronID=np.argmin(np.abs(SpikeTimeE[k,1]-ThetaE))
            SumSinE[NeuronID]+=np.sin(SpikeTimePhaseE[k,2])
            SumCosE[NeuronID]+=np.cos(SpikeTimePhaseE[k,2])
            CountsE[NeuronID]+=1

    MeanSinE=SumSinE/CountsE
    MeanCosE=SumCosE/CountsE
    PhaseLockingIndexE[:,0]=np.arctan2(MeanSinE,MeanCosE)
    PhaseLockingIndexE[:,1]=np.sqrt(MeanCosE**2+MeanSinE**2)

    SpikeTimePhaseI=np.zeros((SpikeTimeI.shape[0],3))
    SpikeTimePhaseI[:,0:2]=SpikeTimeI
    PhaseLockingIndexI=np.zeros((NI,2))
    SumSinI=np.zeros(NI)
    SumCosI=np.zeros(NI)
    CountsI=np.zeros(NI)
    for k in range(SpikeTimeI.shape[0]):
        TbinID=np.argmin(np.abs(SpikeTimeI[k,0]-TauGaba-Tbins))
        SpikeTimePhaseI[k,2]=Phase_SGabaE[TbinID]

        if SpikeTimeI[k,0]>(Stim2EEnd+500):
            NeuronID=np.argmin(np.abs(SpikeTimeI[k,1]-ThetaI))
            SumSinI[NeuronID]+=np.sin(SpikeTimePhaseI[k,2])
            SumCosI[NeuronID]+=np.cos(SpikeTimePhaseI[k,2])
            CountsI[NeuronID]+=1

    MeanSinI=SumSinI/CountsI
    MeanCosI=SumCosI/CountsI
    PhaseLockingIndexI[:,0]=np.arctan2(MeanSinI,MeanCosI)
    PhaseLockingIndexI[:,1]=np.sqrt(MeanCosI**2+MeanSinI**2)
    
    ESpike2SGabaE_PLI=np.zeros(2)
    DeltaPhaseE=Phase_SGabaE[200:]-Phase_ESpikeHist[200:]
    yE=np.mean(np.sin(DeltaPhaseE))
    xE=np.mean(np.cos(DeltaPhaseE))
    ESpike2SGabaE_PLI[0]=np.arctan2(yE,xE)
    ESpike2SGabaE_PLI[1]=np.sqrt(xE**2+yE**2)

    ISpike2SGabaE_PLI=np.zeros(2)
    DeltaPhaseI=Phase_SGabaE[200:]-Phase_ISpikeHist[200:]
    yI=np.mean(np.sin(DeltaPhaseI))
    xI=np.mean(np.cos(DeltaPhaseI))
    ISpike2SGabaE_PLI[0]=np.arctan2(yI,xI)
    ISpike2SGabaE_PLI[1]=np.sqrt(xI**2+yI**2)

    np.savez_compressed(FilePath+'Result_'+str(TrialID)+'.npz',
        Paras=Parameters,ThetaE=ThetaE,ThetaI=ThetaI,
        STPE=SpikeTimePhaseE,STPI=SpikeTimePhaseI,ProfE=ProfE,ProfI=ProfI,PVs=PVs,NormPVs=NormPVs,Tbinedges=Tbinedges,Tbins=Tbins,
        IEE=-CurrentEE/CurrentCount,IIE=-CurrentIE/CurrentCount,INE=-CurrentNE/CurrentCount,ILE=-CurrentLE/CurrentCount,IBE=-CurrentBE,
        IEI=-CurrentEI/CurrentCount,III=-CurrentII/CurrentCount,INI=-CurrentNI/CurrentCount,ILI=-CurrentLI/CurrentCount,IBI=-CurrentBI,IMS=-CurrentMS,
        Freq=Freq,G2E=Gaba2E,SGE=SumGaba2E,PSD_SGE=PSD_SGabaE,
        MPid_SGE=MaxPeakID_PSD_SGabaE,TP_SGE=thetaPower_SGabaE,RTP_SGE=Relative_thetaPower_SGabaE,FTP_SGE=Fraction_thetaPower_SGabaE,T2D_SGE=theta2delta_SGabaE,
        MTPid_SGE=MaxThetaPeakID_PSD_SGabaE,TPP_SGE=thetaPeakPower_SGabaE,RTPP_SGE=Relative_thetaPeakPower_SGabaE,FTPP_SGE=Fraction_thetaPeakPower_SGabaE,
        ESH=ESpikeHist,PSD_ESH=PSD_ESpike,
        MPid_ESH=MaxPeakID_PSD_ESpike,TP_ESH=thetaPower_ESpike,RTP_ESH=Relative_thetaPower_ESpike,FTP_ESH=Fraction_thetaPower_ESpike,T2D_ESH=theta2delta_ESpike,
        MTPid_ESH=MaxThetaPeakID_PSD_ESpike,TPP_ESH=thetaPeakPower_ESpike,RTPP_ESH=Relative_thetaPeakPower_ESpike,FTPP_ESH=Fraction_thetaPeakPower_ESpike,
        ISH=ISpikeHist,PSD_ISH=PSD_ISpike,
        MPid_ISH=MaxPeakID_PSD_ISpike,TP_ISH=thetaPower_ISpike,RTP_ISH=Relative_thetaPower_ISpike,FTP_ISH=Fraction_thetaPower_ISpike,T2D_ISH=theta2delta_ISpike,
        MTPid_ISH=MaxThetaPeakID_PSD_ISpike,TPP_ISH=thetaPeakPower_ISpike,RTPP_ISH=Relative_thetaPeakPower_ISpike,FTPP_ISH=Fraction_thetaPeakPower_ISpike,
        VmE=VmE,PSD_VmE=PSD_VmE,VmE_RecNeu_ID=VmE_RecNeu_ID,
        MPid_VmE=MaxPeakID_PSD_VmE,TP_VmE=thetaPower_VmE,RTP_VmE=Relative_thetaPower_VmE,FTP_VmE=Fraction_thetaPower_VmE,T2D_VmE=theta2delta_VmE,
        MTPid_VmE=MaxThetaPeakID_PSD_VmE,TPP_VmE=thetaPeakPower_VmE,RTPP_VmE=Relative_thetaPeakPower_VmE,FTPP_VmE=Fraction_thetaPeakPower_VmE,
        VmI=VmI,PSD_VmI=PSD_VmI,VmI_RecNeu_ID=VmI_RecNeu_ID,
        MPid_VmI=MaxPeakID_PSD_VmI,TP_VmI=thetaPower_VmI,RTP_VmI=Relative_thetaPower_VmI,FTP_VmI=Fraction_thetaPower_VmI,T2D_VmI=theta2delta_VmI,
        MTPid_PSD_VmI=MaxThetaPeakID_PSD_VmI,TPP_VmI=thetaPeakPower_VmI,RTPP_VmI=Relative_thetaPeakPower_VmI,FTPP_VmI=Fraction_thetaPeakPower_VmI,
        ACbinedges=ACbinedges,ACbins=ACbins,ACEs=ACEs,ACIs=ACIs,
        Filt_SGE=filt_SGabaE,HT_SGE=hilbert_SGabaE,Phase_SGE=Phase_SGabaE,
        Filt_ESH=filt_ESpikeHist,HT_ESH=hilbert_ESpikeHist,Phase_ESpikeHist=Phase_ESpikeHist,
        Filt_ISH=filt_ISpikeHist,HT_ISH=hilbert_ISpikeHist,Phase_ISpikeHist=Phase_ISpikeHist,
        PLIE=PhaseLockingIndexE,PLII=PhaseLockingIndexI,
        ESH2SGE_PLI=ESpike2SGabaE_PLI,ISH2SGE_PLI=ISpike2SGabaE_PLI
        )

    fig=plt.figure(figsize=(15,18))
    ax1=fig.add_axes([0.10,0.83,0.45,0.15])
    ax2=fig.add_axes([0.10,0.36,0.45,0.15])
    ax3=fig.add_axes([0.55,0.83,0.10,0.15])
    ax4=fig.add_axes([0.55,0.36,0.10,0.15])
    ax5=fig.add_axes([0.70,0.83,0.20,0.15])
    ax6=fig.add_axes([0.70,0.36,0.20,0.15])
    ax7a=fig.add_axes([0.10,0.71,0.45,0.08])
    ax7b=fig.add_axes([0.10,0.63,0.45,0.08])
    ax7c=fig.add_axes([0.10,0.55,0.45,0.08])
    ax8a=fig.add_axes([0.10,0.24,0.45,0.08])
    ax8b=fig.add_axes([0.10,0.16,0.45,0.08])
    ax9a=fig.add_axes([0.60,0.71,0.30,0.08])
    ax9b=fig.add_axes([0.60,0.63,0.30,0.08])
    ax9c=fig.add_axes([0.60,0.55,0.30,0.08])
    axXa=fig.add_axes([0.60,0.24,0.30,0.08])
    axXb=fig.add_axes([0.60,0.16,0.30,0.08])
    axY=fig.add_axes([0.10,0.04,0.25,0.08])
    axZ=fig.add_axes([0.40,0.02,0.12,0.10],projection='polar')
    axA=fig.add_axes([0.58,0.02,0.12,0.10],projection='polar')
    axB=fig.add_axes([0.76,0.02,0.12,0.10],projection='polar')

    ShowVmEID=np.where(VmE_RecNeu_ID==512)[0][0]
    ShowVmIID=np.where(VmI_RecNeu_ID==128)[0][0]
    ax1.scatter(SpikeTimePhaseE[:,0],SpikeTimePhaseE[:,1],s=0.5,c='r',marker='.',linewidths=0,label='SpikeTime')
    ax2.scatter(SpikeTimePhaseI[:,0],SpikeTimePhaseI[:,1],s=0.5,c='r',marker='.',linewidths=0,label='SpikeTime')
    ax1.plot(PVs[:,0],PVs[:,1]/3)
    ax1.plot(Tbins,NormPVs[:,0]/3+120,color='tab:green')
    ax3.plot(ProfE,ThetaE)
    ax4.plot(ProfI,ThetaI)
    ax5.plot(-CurrentEE/CurrentCount,ThetaE,label='IEE')
    ax5.plot(-CurrentIE/CurrentCount,ThetaE,label='IIE')
    ax5.plot(-CurrentBE,ThetaE,label='IBE')
    ax5.plot(-CurrentNE/CurrentCount,ThetaE,label='INE')
    ax5.plot((-CurrentEE/CurrentCount)+(-CurrentIE/CurrentCount)+(-CurrentBE)+(-CurrentNE/CurrentCount),ThetaE,label='Itot')
    ax6.plot(-CurrentEI/CurrentCount,ThetaI,label='IEI')
    ax6.plot(-CurrentII/CurrentCount,ThetaI,label='III')
    ax6.plot(-CurrentBI-CurrentMS,ThetaI,label='IBI+IMS')
    ax6.plot(-CurrentNI/CurrentCount,ThetaI,label='INI')
    ax6.plot((-CurrentEI/CurrentCount)+(-CurrentII/CurrentCount)+(-CurrentBI-CurrentMS)+(-CurrentNI/CurrentCount),ThetaI,label='Itot')
    ax5.legend()
    ax6.legend()
    ax7a.plot(Tbins,SumGaba2E)
    ax7b.plot(Tbins,ESpikeHist)
    ax7c.plot(Tbins,VmE[ShowVmEID,:])
    ax7a.plot(Tbins,filt_SGabaE+np.mean(SumGaba2E))
    ax7b.plot(Tbins,filt_ESpikeHist+np.mean(ESpikeHist))
    ax8a.plot(Tbins,ISpikeHist)
    ax8a.plot(Tbins,filt_ISpikeHist+np.mean(ISpikeHist))
    ax8b.plot(Tbins,VmI[ShowVmIID,:])
    ax9a.plot(Freq,PSD_SGabaE)
    ax9b.plot(Freq,PSD_ESpike)
    ax9c.plot(Freq,PSD_VmE[ShowVmEID,:])
    ax9a.scatter(Freq[MaxPeakID_PSD_SGabaE],PSD_SGabaE[MaxPeakID_PSD_SGabaE]*1.2,marker='x')
    ax9a.scatter(Freq[MaxThetaPeakID_PSD_SGabaE],PSD_SGabaE[MaxPeakID_PSD_SGabaE]*1.1,marker='v')
    ax9b.scatter(Freq[MaxPeakID_PSD_ESpike],PSD_ESpike[MaxPeakID_PSD_ESpike]*1.2,marker='x')
    ax9b.scatter(Freq[MaxThetaPeakID_PSD_ESpike],PSD_ESpike[MaxPeakID_PSD_ESpike]*1.1,marker='v')
    ax9c.scatter(Freq[MaxPeakID_PSD_VmE[ShowVmEID]],PSD_VmE[ShowVmEID,MaxPeakID_PSD_VmE[ShowVmEID]]*1.2,marker='x')
    ax9c.scatter(Freq[MaxThetaPeakID_PSD_VmE[ShowVmEID]],PSD_VmE[ShowVmEID,MaxPeakID_PSD_VmE[ShowVmEID]]*1.1,marker='v')
    ax9a.vlines(4,0,np.max(PSD_SGabaE),color='k',linestyle='--')
    ax9a.vlines(12,0,np.max(PSD_SGabaE),color='k',linestyle='--')
    ax9b.vlines(4,0,np.max(PSD_ESpike),color='k',linestyle='--')
    ax9b.vlines(12,0,np.max(PSD_ESpike),color='k',linestyle='--')
    ax9c.vlines(4,0,np.max(PSD_VmE[ShowVmEID,:]),color='k',linestyle='--')
    ax9c.vlines(12,0,np.max(PSD_VmE[ShowVmEID,:]),color='k',linestyle='--')
    axXa.plot(Freq,PSD_ISpike)
    axXb.plot(Freq,PSD_VmI[ShowVmIID,:])
    axXa.scatter(Freq[MaxPeakID_PSD_ISpike],PSD_ISpike[MaxPeakID_PSD_ISpike]*1.2,marker='x')
    axXa.scatter(Freq[MaxThetaPeakID_PSD_ISpike],PSD_ISpike[MaxPeakID_PSD_ISpike]*1.1,marker='v')
    axXb.scatter(Freq[MaxPeakID_PSD_VmI[ShowVmIID]],PSD_VmI[ShowVmIID,MaxPeakID_PSD_VmI[ShowVmIID]]*1.2,marker='x')
    axXb.scatter(Freq[MaxThetaPeakID_PSD_VmI[ShowVmIID]],PSD_VmI[ShowVmIID,MaxPeakID_PSD_VmI[ShowVmIID]]*1.1,marker='v')
    axXa.vlines(4,0,np.max(PSD_ISpike),color='k',linestyle='--')
    axXa.vlines(12,0,np.max(PSD_ISpike),color='k',linestyle='--')
    axXb.vlines(4,0,np.max(PSD_VmI[ShowVmIID,:]),color='k',linestyle='--')
    axXb.vlines(12,0,np.max(PSD_VmI[ShowVmIID,:]),color='k',linestyle='--')
    axY.plot(ACbins[5:],ACEs[512,5:])
    axY.plot(ACbins[5:],ACIs[128,5:])
    axZ.scatter(PhaseLockingIndexE[512-64:512+64,0],PhaseLockingIndexE[512-64:512+64,1])
    axA.scatter(PhaseLockingIndexI[128-16:128+16,0],PhaseLockingIndexI[128-16:128+16,1])
    axB.scatter(ESpike2SGabaE_PLI[0],ESpike2SGabaE_PLI[1])
    axB.scatter(ISpike2SGabaE_PLI[0],ISpike2SGabaE_PLI[1])
    ax1.set_xlim(0,T)
    ax1.set_ylim(0,360)
    ax1.set_xticks(np.arange(int(T/1000)+1)*1000)
    ax1.set_xticklabels(np.arange(int(T/1000)+1))
    ax1.set_yticks([0,180,360])
    ax2.set_xlim(0,T)
    ax2.set_ylim(0,360)
    ax2.set_xticks(np.arange(int(T/1000)+1)*1000)
    ax2.set_xticklabels(np.arange(int(T/1000)+1))
    ax2.set_yticks([0,180,360])
    ax3.set_xlim(0,np.max(ProfE)*1.1)
    ax3.set_ylim(0,360)
    ax3.set_yticks([0,180,360])
    ax3.set_yticklabels([])
    ax4.set_xlim(0,np.max(ProfI)*1.1)
    ax4.set_ylim(0,360)
    ax4.set_yticks([0,180,360])
    ax4.set_yticklabels([])
    ax5.set_ylim(0,360)
    ax5.set_yticks([0,180,360])
    ax6.set_ylim(0,360)
    ax6.set_yticks([0,180,360])
    ax7a.set_xlim(0,T)
    ax7b.set_xlim(0,T)
    ax7c.set_xlim(0,T)
    ax7a.set_xticks(np.arange(int(T/1000)+1)*1000)
    ax7a.set_xticklabels(np.arange(int(T/1000)+1))
    ax7b.set_xticks(np.arange(int(T/1000)+1)*1000)
    ax7b.set_xticklabels(np.arange(int(T/1000)+1))
    ax7c.set_xticks(np.arange(int(T/1000)+1)*1000)
    ax7c.set_xticklabels(np.arange(int(T/1000)+1))
    ax8a.set_xlim(0,T)
    ax8a.set_xticks(np.arange(int(T/1000)+1)*1000)
    ax8a.set_xticklabels(np.arange(int(T/1000)+1))
    ax8b.set_xlim(0,T)
    ax8b.set_xticks(np.arange(int(T/1000)+1)*1000)
    ax8b.set_xticklabels(np.arange(int(T/1000)+1))
    ax9a.set_xlim(0,50)
    ax9b.set_xlim(0,50)
    ax9c.set_xlim(0,50)
    axXa.set_xlim(0,50)
    axXb.set_xlim(0,50)
    axY.set_xlim(0,1)
    plt.savefig(FilePath+'Preview_'+str(TrialID)+'.png',format='PNG',dpi=150)
    # plt.show()
    plt.close()

if __name__=='__main__':
    main(TrialID)
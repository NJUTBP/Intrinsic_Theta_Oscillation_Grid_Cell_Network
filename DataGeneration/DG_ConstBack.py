# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 2021

@author: Tao WANG

Description: Generating data for Figure 3a-d in main text.
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

#### simulation settings ####
DeltaT=0.02#ms
T=3500.0#ms

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
sigmaEI=30.0#degree
#### IE connection profile ####
muIE=90.0
sigmaIE=30.0#degree
#### magnesium ion ####
Mg=1.0
#### synaptic conductance ####
GNmdaI=0.584
GGabaE=1.077
GGabaI=2.048

#### time constant for synapses ####
TauAmpa=2.0#ms
global TauGaba
TauGaba=10#ms
TauX=2.0#ms
global TauNmda
TauNmda=100#ms
AlphaNmda=0.5#kHz
global MultiInput
MultiInput=1.0

#### background parameters ####
# Fext=1.8#float(sys.argv[1])#kHz
# GExtE=3.1#nS
# GExtI=2.38#nS

#### stimulus ####
# Tcuestart=0.0#ms
# Tcueend=250.0#ms
# IcueMax=0.0#pA
# Icue=np.append(IcueMax*np.ones(51),np.zeros(NE-102))#pA
# Icue=np.append(Icue,IcueMax*np.ones(51))#pA
# # Icue=np.append(np.zeros(int(NE/2-51)),IcueMax*np.ones(102))#pA
# # Icue=np.append(Icue,np.zeros(int(NE/2-51)))#pA

#### saving settings ####
SavePath='../Data/DG_ConstBack/IE1600II1400'
isExist=os.path.exists(SavePath)
if not isExist: os.makedirs(SavePath)
np.savetxt(SavePath+'/Parameters_'+str(TrialID)+'.txt',np.array([DeltaT,T,NE,NI,CME,CMI,ZE,ZI,tauref_E,tauref_I,GLE,GLI,ZL,Eth,Eres,JEIpos,sigmaEI,muIE,sigmaIE,Mg,GNmdaI,GGabaE,GGabaI,TauAmpa,TauGaba,TauX,TauNmda,AlphaNmda]))#,Fext,GExtE,GExtI]))

def main():
    #### several values ####
    Factor=1.0
    DecX=np.exp(-DeltaT/TauX)
    DecA=np.exp(-DeltaT/TauAmpa)
    DecG=np.exp(-DeltaT/TauGaba)
    DecXh=np.exp(-DeltaT/(2*TauX))
    DecAh=np.exp(-DeltaT/(2*TauAmpa))
    DecGh=np.exp(-DeltaT/(2*TauGaba))
    #### initializing excitatory cells ####
    # VE=np.random.uniform(-60.0,-50.0,NE)
    VE=-67.0*np.ones(NE)
    ThetaE=np.linspace(0,360,NE,endpoint=False)
    RefCountsE=(Maxref_E*np.ones(NE)).astype(np.int32)
    #### initializing interneurons ####
    # VI=np.random.uniform(-60.0,-50.0,NI)
    VI=-67.0*np.ones(NI)
    ThetaI=np.linspace(0,360,NI,endpoint=False)
    RefCountsI=(Maxref_I*np.ones(NI)).astype(np.int32)
    
    #### defining WEI ####
    IntegratePoints=np.linspace(0,180,10000)
    SumExp=np.sum(np.exp(-IntegratePoints**2/(2*sigmaEI**2)))
    JEIneg=(10000-SumExp*JEIpos)/(10000-SumExp)
    DeltaTheta=np.abs(ThetaE-ThetaI[0])*np.append(np.ones(int(NE/2)),np.zeros(int(NE/2)))+np.abs(360-ThetaE+ThetaI[0])*np.append(np.zeros(int(NE/2)),np.ones(int(NE/2)))
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

    #### simulation ####
    STEPs=int(T/DeltaT)
    SingleVE=np.zeros(int(STEPs/10))
    SingleVI=np.zeros(int(STEPs/10))
    CurrentEI=np.zeros(int(STEPs/10))
    CurrentIE=np.zeros(int(STEPs/10))
    for i in range(STEPs):
        # if i%1000==0:
        #     print("Please wait... %.2f%% Finished!" % (i*100/STEPs))
        
        #### adding stimulus ####
        Istim=0.0
        # if Tcuestart<(i*DeltaT) and (i*DeltaT)<Tcueend:
        #     Istim=Icue
        # else:
        #     Istim=0.0
        
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
        # IextE=(VE-ZE)*GExtE*SExtE
        IextE=-1600*Factor*MultiInput
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
        # IextI=(VI-ZE)*GExtI*SExtI
        IextI=-1400*Factor*MultiInput
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
        # IextE=(VEtemp-ZE)*GExtE*SExtETemp
        IextE=-1600*Factor*MultiInput
        #### calculating leaky currents ####
        IleakE=GLE*(VEtemp-ZL)
        #### integration ####
        K2E=(Istim-IGabaE-IextE-IleakE)/CME
        VE=VE+(DeltaT/2)*(K1E+K2E)*OutrefE

        ######## interneurons ########
        ### calculating Nmda EI currents ####
        FSNmdaTemp=np.fft.fft(SNmdaTemp)
        FWEI_SNmdaTemp=FWEI*FSNmdaTemp
        SumNEITemp=np.real(np.fft.ifft(FWEI_SNmdaTemp))[0::4]
        INmdaI=(VItemp-ZE)*GNmdaI*SumNEITemp/(1.0+Mg*np.exp(-0.062*VItemp)/3.57)
        #### calculating Gaba II currents ####
        SumGIITemp=np.sum(SGabaTemp)
        # SumGIITemp=SumGII*DecGh
        IGabaI=(VItemp-ZI)*GGabaI*SumGIITemp
        #### calculating background currents ####
        # IextI=(VItemp-ZE)*GExtI*SExtITemp
        IextI=-1400*Factor*MultiInput
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
        # FlagB2E=np.random.poisson((Fext*DeltaT),NE)
        # FlagB2I=np.random.poisson((Fext*DeltaT),NI)
        # SExtE+=FlagB2E
        # SExtI+=FlagB2I

        ######## record ########
        for Eid in np.where(FlagE)[0]:
            SpikeTimeE=np.append(SpikeTimeE,np.array([[(i*DeltaT),ThetaE[Eid]]]),axis=0)
        for Iid in np.where(FlagI)[0]:
            SpikeTimeI=np.append(SpikeTimeI,np.array([[(i*DeltaT),ThetaI[Iid]]]),axis=0)
        if i%10==0:
            SingleVE[int(i/10)]=VE[512]
            SingleVI[int(i/10)]=VI[128]
            CurrentEI[int(i/10)]=INmdaI[128]
            CurrentIE[int(i/10)]=IGabaE[512]

        ############ end of this step ############

    #### save initial recordings ####
    np.savez_compressed(SavePath+'/SpikeTimeE_MI'+str(MultiInput)+'_TN'+str(TauNmda)+'_TG'+str(TauGaba)+'.npz',SpikeTimeE)
    np.savez_compressed(SavePath+'/SpikeTimeI_MI'+str(MultiInput)+'_TN'+str(TauNmda)+'_TG'+str(TauGaba)+'.npz',SpikeTimeI)
    np.savez_compressed(SavePath+'/SingleVE_MI'+str(MultiInput)+'_TN'+str(TauNmda)+'_TG'+str(TauGaba)+'.npz',SingleVE)
    np.savez_compressed(SavePath+'/SingleVI_MI'+str(MultiInput)+'_TN'+str(TauNmda)+'_TG'+str(TauGaba)+'.npz',SingleVI)
    np.savez_compressed(SavePath+'/CurrentEI_MI'+str(MultiInput)+'_TN'+str(TauNmda)+'_TG'+str(TauGaba)+'.npz',CurrentEI)
    np.savez_compressed(SavePath+'/CurrentIE_MI'+str(MultiInput)+'_TN'+str(TauNmda)+'_TG'+str(TauGaba)+'.npz',CurrentIE)

    hist,binedges=np.histogram(SpikeTimeE[:,0],bins=np.linspace(0,3500,350,endpoint=False))

    f, Pxx_den = signal.welch(hist, fs=100, window=signal.get_window('hamming',128), noverlap=64, nfft=1024)

    # Mask=(2<f)*(f<=6)
    # print(f[np.where(Pxx_den*Mask==np.max(Pxx_den*Mask))[0][0]])
    # np.savetxt('Data/Figure5/Freq_MI'+str(MultiInput)+'_TN'+str(TauNmda)+'_TG'+str(TauGaba)+'.txt',f[np.where(Pxx_den*Mask==np.max(Pxx_den*Mask))[0]])

    #### save PSD analysis ####
    print(f[np.where(Pxx_den==np.max(Pxx_den))[0][0]])
    np.savetxt(SavePath+'/Freq_MI'+str(MultiInput)+'_TN'+str(TauNmda)+'_TG'+str(TauGaba)+'.txt',f[np.where(Pxx_den==np.max(Pxx_den))[0]])

    # fig=plt.figure(figsize=(9,9))
    # ax1=fig.add_subplot(411)
    # ax2=fig.add_subplot(412)
    # ax3=fig.add_subplot(413)
    # ax4=fig.add_subplot(414)
    # ax1.scatter(SpikeTimeE[:,0],SpikeTimeE[:,1],s=0.5,c='r',marker='.',linewidths=0,label='SpikeTime')
    # ax2.scatter(SpikeTimeI[:,0],SpikeTimeI[:,1],s=0.5,c='b',marker='.',linewidths=0,label='SpikeTime')
    # ax3.plot(np.linspace(0,3500,SingleVE.size,endpoint=False),SingleVE,color='steelblue')
    # ax3.plot(np.linspace(0,3500,SingleVI.size,endpoint=False),SingleVI-18,color='darkorange')
    # ax4.plot(f[1:-1],10*np.log10(Pxx_den[1:-1]))
    # ax1.set_xlim((0,3500))
    # ax1.set_ylim((0,360))
    # ax2.set_xlim((0,3500))
    # ax2.set_ylim((0,360))
    # ax3.set_xlim((0,3500))
    # ax2.set_ylim((0,360))
    # ax1.set_xticks(np.arange(0,3501,1000))
    # ax1.set_yticks(np.arange(0,361,180))
    # ax2.set_xticks(np.arange(0,3501,1000))
    # ax2.set_yticks(np.arange(0,361,180))
    # ax3.set_xticks(np.arange(0,3501,1000))
    # plt.savefig(SavePath+'/SpikeTimeE_MI'+str(MultiInput)+'_TN'+str(TauNmda)+'_TG'+str(TauGaba)+'.png', format='png', dpi=300)
    # # plt.show()
    # plt.close()

if __name__=='__main__':
    # main()
    MultiInput=1.0
    TauNmda=100
    for k in [5,6,7,8,9,10,11,12,13,14,15]:#
        TauGaba=k
        main()

    # TauNmda=100
    # TauGaba=10
    # for i in [0.5,1.0,1.5,2.0,2.5,3.0]:
    #     MultiInput=i
    #     main()

    MultiInput=1.0
    TauGaba=10
    for j in [50,60,70,80,90,100,110,120,130,140,150]:
        TauNmda=j
        main()

# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 2021

@author: Tao WANG

Description: Generating data for Figure S5 in main text.
"""

####importing####
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import ode
import scipy.linalg as sl
from scipy.signal import convolve

def RK4(taus,gamma,je,ji,i0,a,b,d,T,dt,s1,s2):
    result=[]
    time=[]
    for i in range(int(T/dt)):
        i1=je*s1-ji*s2+i0
        i2=je*s2-ji*s1+i0
        r1=(a*i1-b)/(1-np.exp(-d*(a*i1-b)))
        r2=(a*i2-b)/(1-np.exp(-d*(a*i2-b)))

        time.append(i*dt)
        result.append([s1,s2,r1,r2])

        k1s1=-s1/taus+(1-s1)*gamma*r1
        k1s2=-s2/taus+(1-s2)*gamma*r2

        i1=je*(s1+k1s1*dt/2)-ji*(s2+k1s2*dt/2)+i0
        i2=je*(s2+k1s2*dt/2)-ji*(s1+k1s1*dt/2)+i0
        r1=(a*i1-b)/(1-np.exp(-d*(a*i1-b)))
        r2=(a*i2-b)/(1-np.exp(-d*(a*i2-b)))

        k2s1=-(s1+k1s1*dt/2)/taus+(1-(s1+k1s1*dt/2))*gamma*r1
        k2s2=-(s2+k1s2*dt/2)/taus+(1-(s2+k1s2*dt/2))*gamma*r2

        i1=je*(s1+k2s1*dt/2)-ji*(s2+k2s2*dt/2)+i0
        i2=je*(s2+k2s2*dt/2)-ji*(s1+k2s1*dt/2)+i0
        r1=(a*i1-b)/(1-np.exp(-d*(a*i1-b)))
        r2=(a*i2-b)/(1-np.exp(-d*(a*i2-b)))

        k3s1=-(s1+k2s1*dt/2)/taus+(1-(s1+k2s1*dt/2))*gamma*r1
        k3s2=-(s2+k2s2*dt/2)/taus+(1-(s2+k2s2*dt/2))*gamma*r2

        i1=je*(s1+k3s1*dt)-ji*(s2+k3s2*dt)+i0
        i2=je*(s2+k3s2*dt)-ji*(s1+k3s1*dt)+i0
        r1=(a*i1-b)/(1-np.exp(-d*(a*i1-b)))
        r2=(a*i2-b)/(1-np.exp(-d*(a*i2-b)))

        k4s1=-(s1+k3s1*dt)/taus+(1-(s1+k3s1*dt))*gamma*r1
        k4s2=-(s2+k3s2*dt)/taus+(1-(s2+k3s2*dt))*gamma*r2

        s1=s1+dt*(k1s1+2*k2s1+2*k3s1+k4s1)/6
        s2=s2+dt*(k1s2+2*k2s2+2*k3s2+k4s2)/6

    time=np.array(time)
    result=np.array(result)

    return time,result

def vecfld(taus,gamma,je,ji,i0,a,b,d):
    s2,s1=np.mgrid[0:1:1001j,0:1:1001j]

    i1=je*s1-ji*s2+i0
    i2=je*s2-ji*s1+i0
    r1=(a*i1-b)/(1-np.exp(-d*(a*i1-b)))
    r2=(a*i2-b)/(1-np.exp(-d*(a*i2-b)))

    ks1=-s1/taus+(1-s1)*gamma*r1
    ks2=-s2/taus+(1-s2)*gamma*r2

    spd=np.sqrt(ks1**2+ks2**2)

    return s1,s2,ks1,ks2,spd

def main_0():
    ####Parameters####
    taus=0.1
    gamma=0.641
    a=270
    b=108
    d=0.154
    ####Excitatory####
    ##Original##
    # je=0.2609
    # ji=0.0497
    # i0=0.3255
    ##Modified####
    je=0.0
    ji=0.32
    i0=0.60
    ####inhibitory####
    # je=0.0
    # ji=0.32
    # i0=0.60
    ####initialization####
    s1=0.5
    s2=0
    T=5
    dt=0.001

    time,result=RK4(taus,gamma,je,ji,i0,a,b,d,T,dt,s1,s2)

    s1,s2,ks1,ks2,spd=vecfld(taus,gamma,je,ji,i0,a,b,d)

    #### Nucline ####
    nullcline_point_index_s1=np.where(np.abs(ks1)<5e-3)
    nullcline_point_index_s2=np.where(np.abs(ks2)<5e-3)
    argsort_nullcline_point_index_s1=np.argsort(s2[nullcline_point_index_s1[1],0])
    argsort_nullcline_point_index_s2=np.argsort(s1[0,nullcline_point_index_s2[0]])

    #### theoretical Nucline in pure inhibition ####
    nullcline1_s1=np.linspace(0,1,1001)
    nullcline1_s2=(taus*gamma*((a*(-ji*nullcline1_s1+i0)-b)/(1-np.exp(-d*(a*(-ji*nullcline1_s1+i0)-b)))))/(1+taus*gamma*((a*(-ji*nullcline1_s1+i0)-b)/(1-np.exp(-d*(a*(-ji*nullcline1_s1+i0)-b)))))
    nullcline2_s2=np.linspace(0,1,1001)
    nullcline2_s1=(taus*gamma*((a*(-ji*nullcline2_s2+i0)-b)/(1-np.exp(-d*(a*(-ji*nullcline2_s2+i0)-b)))))/(1+taus*gamma*((a*(-ji*nullcline2_s2+i0)-b)/(1-np.exp(-d*(a*(-ji*nullcline2_s2+i0)-b)))))

    TNcL1=np.append(nullcline1_s1.reshape((1001,1)),nullcline1_s2.reshape((1001,1)),axis=1)
    TNcL2=np.append(nullcline2_s1.reshape((1001,1)),nullcline2_s2.reshape((1001,1)),axis=1)

    ####find fix point####
    #### The following codes have problems, but don't affect the results ####
    islocalmin=np.zeros((1001,1001)).astype(np.bool)
    zone=10
    for i in range(zone,1001-zone):
        for j in range(zone,1001-zone):
            if spd[i,j]==np.min(spd[i-zone:i+1+zone,j-zone:j+1+zone]):
                # print(spd[i-zone:i+1+zone,j-zone:j+1+zone])
                islocalmin[i,j]=True
    NcL1=np.append(s1[0,nullcline_point_index_s1[0]][argsort_nullcline_point_index_s1].reshape((s1[0,nullcline_point_index_s1[0]][argsort_nullcline_point_index_s1].size,1)),s2[nullcline_point_index_s1[1],0][argsort_nullcline_point_index_s1].reshape((s2[nullcline_point_index_s1[1],0][argsort_nullcline_point_index_s1].size,1)),axis=1)
    NcL2=np.append(s1[0,nullcline_point_index_s2[0]][argsort_nullcline_point_index_s2].reshape((s1[0,nullcline_point_index_s2[0]][argsort_nullcline_point_index_s2].size,1)),s2[nullcline_point_index_s2[1],0][argsort_nullcline_point_index_s2].reshape((s2[nullcline_point_index_s2[1],0][argsort_nullcline_point_index_s2].size,1)),axis=1)
    FP=np.append(s1[0,np.where(islocalmin)[0]].reshape((s1[0,np.where(islocalmin)[0]].size,1)),s2[np.where(islocalmin)[1],0].reshape((s2[np.where(islocalmin)[1],0].size,1)),axis=1)
    if FP.shape[0]%2==0:
        FProw=FP.shape[0]
        NewFP=np.append(FP[:int(FProw/2)-1,:],(FP[int(FProw/2)-1,:]+FP[int(FProw/2),:]).reshape((1,2))/2,axis=0)
        NewFP=np.append(NewFP,FP[(int(FProw/2)+1):,:],axis=0)
        FP=NewFP
    if je>0:
        np.savez_compressed('../Data/DG_ReducedModel/Excitatory.npz',s1=s1,s2=s2,ks1=ks1,ks2=ks2,spd=spd,NcL1=NcL1,NcL2=NcL2,FP=FP)
    else:
        np.savez_compressed('../Data/DG_ReducedModel/Inhibitory.npz',s1=s1,s2=s2,ks1=ks1,ks2=ks2,spd=spd,NcL1=NcL1,NcL2=NcL2,TNcL1=TNcL1,TNcL2=TNcL2,FP=FP)

    fig=plt.figure(figsize=(18,6))
    ax1=fig.add_subplot(131)
    ax2=fig.add_subplot(132)
    ax3=fig.add_subplot(133)

    ax1.plot(time,result[:,2])
    ax1.plot(time,result[:,3])

    ax2.streamplot(s1[0,:], s2[:,0], ks1, ks2, color=spd, linewidth=0.5, density=[2,3], cmap=plt.cm.autumn)
    ax2.scatter(s1[0,nullcline_point_index_s1[0]][argsort_nullcline_point_index_s1], s2[nullcline_point_index_s1[1],0][argsort_nullcline_point_index_s1])
    ax2.scatter(s1[0,nullcline_point_index_s2[0]][argsort_nullcline_point_index_s2], s2[nullcline_point_index_s2[1],0][argsort_nullcline_point_index_s2])
    ax2.plot(NcL1[:,0],NcL1[:,1])
    ax2.plot(NcL2[:,0],NcL2[:,1])
    ax2.scatter(s1[0,np.where(islocalmin)[0]],s2[np.where(islocalmin)[1],0])
    ax2.set_xlabel('s1')
    ax2.set_ylabel('s2')
    ax2.set_xlim(0,1)
    ax2.set_ylim(0,1)

    ax3.plot(nullcline1_s1, nullcline1_s2)
    ax3.plot(nullcline2_s1, nullcline2_s2)
    ax3.set_xlabel('s1')
    ax3.set_ylabel('s2')
    ax3.set_xlim(0,1)
    ax3.set_ylim(0,1)
    if je>0:
        plt.savefig('../Data/DG_ReducedModel/Excitatory.png', fmt='PNG',dpi=300)
    else:
        plt.savefig('../Data/DG_ReducedModel/Inhibitory.png', fmt='PNG',dpi=300)
    plt.show()
    plt.close()

def main_1(trial,je,ji,i0):
    ####Parameters####
    taus=0.1
    gamma=0.641
    a=270
    b=108
    d=0.154
    ####Excitatory####
    ##Original##
    # je=0.2609
    # ji=0.0497
    # i0=0.3255
    ##Modified####
    # je=0.36
    # ji=0.32
    # i0=0.58
    ####inhibitory####
    # je=0.0
    # ji=0.32
    # i0=0.60
    ####initialization####
    s1=0.5
    s2=0
    T=5
    dt=0.001

    time,result=RK4(taus,gamma,je,ji,i0,a,b,d,T,dt,s1,s2)

    s1,s2,ks1,ks2,spd=vecfld(taus,gamma,je,ji,i0,a,b,d)
    ###Nullclines with excitatory connection####
    # print(np.min(np.abs(ks1)),np.max(np.abs(ks1)),np.min(np.abs(ks2)),np.max(np.abs(ks2)))
    nullcline_point_index_s1=np.where(np.abs(ks1)<1e-3)
    nullcline_point_index_s2=np.where(np.abs(ks2)<1e-3)

    argsort_nullcline_point_index_s1=np.argsort(s2[nullcline_point_index_s1[1],0])
    argsort_nullcline_point_index_s2=np.argsort(s1[0,nullcline_point_index_s2[0]])

    ###Nullclines without excitatory connection####
    nullcline1_s1=np.linspace(0,1,1001)
    nullcline1_s2=(taus*gamma*((a*(-ji*nullcline1_s1+i0)-b)/(1-np.exp(-d*(a*(-ji*nullcline1_s1+i0)-b)))))/(1+taus*gamma*((a*(-ji*nullcline1_s1+i0)-b)/(1-np.exp(-d*(a*(-ji*nullcline1_s1+i0)-b)))))

    nullcline2_s2=np.linspace(0,1,1001)
    nullcline2_s1=(taus*gamma*((a*(-ji*nullcline2_s2+i0)-b)/(1-np.exp(-d*(a*(-ji*nullcline2_s2+i0)-b)))))/(1+taus*gamma*((a*(-ji*nullcline2_s2+i0)-b)/(1-np.exp(-d*(a*(-ji*nullcline2_s2+i0)-b)))))

    # fig1,ax1=plt.subplots()
    # ax1.plot(time,result[:,2])
    # ax1.plot(time,result[:,3])
    # # plt.show()

    # fig2,ax2=plt.subplots()
    # strm=ax2.streamplot(s1[0,:], s2[:,0], ks1, ks2, color=spd, linewidth=0.5, density=[2,3], cmap=plt.cm.autumn)
    # ax2.plot(s1[0,nullcline_point_index_s1[0]][argsort_nullcline_point_index_s1], s2[nullcline_point_index_s1[1],0][argsort_nullcline_point_index_s1])
    # ax2.plot(s1[0,nullcline_point_index_s2[0]][argsort_nullcline_point_index_s2], s2[nullcline_point_index_s2[1],0][argsort_nullcline_point_index_s2])
    # ax2.set_xlabel('s1')
    # ax2.set_ylabel('s2')
    # ax2.set_xlim(0,1)
    # ax2.set_ylim(0,1)
    # # plt.show()

    # fig3,ax3=plt.subplots()
    # ax3.plot(nullcline1_s1, nullcline1_s2)
    # ax3.plot(nullcline2_s1, nullcline2_s2)
    # ax3.set_xlabel('s1')
    # ax3.set_ylabel('s2')
    # ax3.set_xlim(0,1)
    # ax3.set_ylim(0,1)
    # plt.show()

    fig=plt.figure(figsize=(18,6))
    ax1=fig.add_subplot(131)
    ax2=fig.add_subplot(132)
    ax3=fig.add_subplot(133)

    ax1.plot(time,result[:,2])
    ax1.plot(time,result[:,3])

    ax2.streamplot(s1[0,:], s2[:,0], ks1, ks2, color=spd, linewidth=0.5, density=[2,3], cmap=plt.cm.autumn)
    ax2.plot(s1[0,nullcline_point_index_s1[0]][argsort_nullcline_point_index_s1], s2[nullcline_point_index_s1[1],0][argsort_nullcline_point_index_s1])
    ax2.plot(s1[0,nullcline_point_index_s2[0]][argsort_nullcline_point_index_s2], s2[nullcline_point_index_s2[1],0][argsort_nullcline_point_index_s2])
    ax2.set_xlabel('s1')
    ax2.set_ylabel('s2')
    ax2.set_xlim(0,1)
    ax2.set_ylim(0,1)

    ax3.plot(nullcline1_s1, nullcline1_s2)
    ax3.plot(nullcline2_s1, nullcline2_s2)
    ax3.set_xlabel('s1')
    ax3.set_ylabel('s2')
    ax3.set_xlim(0,1)
    ax3.set_ylim(0,1)
    plt.show()
    # plt.savefig('Data/Test/%d_%.4f_%.4f_%.4f.png' % (trial,je,ji,i0), fmt='PNG',dpi=150)
    plt.close()

def main_2(trial,je):
    ####Parameters####
    taus=0.1
    gamma=0.641
    a=270
    b=108
    d=0.154
    ####Excitatory####
    ##Original##
    # je=0.2609
    # ji=0.0497
    # i0=0.3255
    ##Modified####
    # je=0.36
    ji=0.32
    i0=0.60
    ####inhibitory####
    # je=0.0
    # ji=0.32
    # i0=0.58
    ####initialization####
    s1=0.5
    s2=0
    T=5
    dt=0.001

    time,result=RK4(taus,gamma,je,ji,i0,a,b,d,T,dt,s1,s2)

    s1,s2,ks1,ks2,spd=vecfld(taus,gamma,je,ji,i0,a,b,d)
    ###Nullclines with excitatory connection####
    # print(np.min(np.abs(ks1)),np.max(np.abs(ks1)),np.min(np.abs(ks2)),np.max(np.abs(ks2)))
    nullcline_point_index_s1=np.where(np.abs(ks1)<1e-3)
    nullcline_point_index_s2=np.where(np.abs(ks2)<1e-3)

    argsort_nullcline_point_index_s1=np.argsort(s2[nullcline_point_index_s1[1],0])
    argsort_nullcline_point_index_s2=np.argsort(s1[0,nullcline_point_index_s2[0]])

    islocalmin=np.zeros((1001,1001)).astype(np.bool)
    zone=10
    for i in range(zone,1001-zone):
        for j in range(zone,1001-zone):
            if spd[i,j]==np.min(spd[i-zone:i+1+zone,j-zone:j+1+zone]):
                # print(spd[i-zone:i+1+zone,j-zone:j+1+zone])
                islocalmin[i,j]=True
    # print(np.where(islocalmin))

    ###Nullclines without excitatory connection####
    # nullcline1_s1=np.linspace(0,1,1001)
    # nullcline1_s2=(taus*gamma*((a*(-ji*nullcline1_s1+i0)-b)/(1-np.exp(-d*(a*(-ji*nullcline1_s1+i0)-b)))))/(1+taus*gamma*((a*(-ji*nullcline1_s1+i0)-b)/(1-np.exp(-d*(a*(-ji*nullcline1_s1+i0)-b)))))

    # nullcline2_s2=np.linspace(0,1,1001)
    # nullcline2_s1=(taus*gamma*((a*(-ji*nullcline2_s2+i0)-b)/(1-np.exp(-d*(a*(-ji*nullcline2_s2+i0)-b)))))/(1+taus*gamma*((a*(-ji*nullcline2_s2+i0)-b)/(1-np.exp(-d*(a*(-ji*nullcline2_s2+i0)-b)))))

    # print(np.where(np.isclose(nullcline1_s1,nullcline2_s1,atol=1e-2)))
    # print(np.where(np.isclose(nullcline1_s2,nullcline2_s2,atol=1e-2)))
    NcL1=np.append(s1[0,nullcline_point_index_s1[0]][argsort_nullcline_point_index_s1].reshape((s1[0,nullcline_point_index_s1[0]][argsort_nullcline_point_index_s1].size,1)),s2[nullcline_point_index_s1[1],0][argsort_nullcline_point_index_s1].reshape((s2[nullcline_point_index_s1[1],0][argsort_nullcline_point_index_s1].size,1)),axis=1)
    NcL2=np.append(s1[0,nullcline_point_index_s2[0]][argsort_nullcline_point_index_s2].reshape((s1[0,nullcline_point_index_s2[0]][argsort_nullcline_point_index_s2].size,1)),s2[nullcline_point_index_s2[1],0][argsort_nullcline_point_index_s2].reshape((s2[nullcline_point_index_s2[1],0][argsort_nullcline_point_index_s2].size,1)),axis=1)
    FP=np.append(s1[0,np.where(islocalmin)[0]].reshape((s1[0,np.where(islocalmin)[0]].size,1)),s2[np.where(islocalmin)[1],0].reshape((s2[np.where(islocalmin)[1],0].size,1)),axis=1)
    # print(NcL1.shape,NcL2.shape,FP.shape)

    np.savez_compressed('../Data/DG_ReducedModel/ReducedFullData/%d_%.3f.npz' % (trial,je),NcL1=NcL1,NcL2=NcL2,FP=FP)

    fig=plt.figure(figsize=(12,6))
    ax1=fig.add_subplot(121)
    ax2=fig.add_subplot(122)
    # ax3=fig.add_subplot(133)

    ax1.plot(time,result[:,2])
    ax1.plot(time,result[:,3])

    ax2.streamplot(s1[0,:], s2[:,0], ks1, ks2, color=spd, linewidth=0.5, density=[2,3], cmap=plt.cm.autumn)
    # ax2.quiver(s1[::100,::100], s2[::100,::100], ks1[::100,::100], ks2[::100,::100])
    ax2.scatter(s1[0,nullcline_point_index_s1[0]][argsort_nullcline_point_index_s1], s2[nullcline_point_index_s1[1],0][argsort_nullcline_point_index_s1])
    ax2.scatter(s1[0,nullcline_point_index_s2[0]][argsort_nullcline_point_index_s2], s2[nullcline_point_index_s2[1],0][argsort_nullcline_point_index_s2])
    ax2.plot(NcL1[:,0],NcL1[:,1])
    ax2.plot(NcL2[:,0],NcL2[:,1])
    ax2.scatter(s1[0,np.where(islocalmin)[0]],s2[np.where(islocalmin)[1],0])
    ax2.set_xlabel('s1')
    ax2.set_ylabel('s2')
    ax2.set_xlim(0,1)
    ax2.set_ylim(0,1)

    # ax3.scatter(nullcline1_s1, nullcline1_s2)
    # ax3.scatter(nullcline2_s1, nullcline2_s2)
    # ax3.set_xlabel('s1')
    # ax3.set_ylabel('s2')
    # ax3.set_xlim(0,1)
    # ax3.set_ylim(0,1)
    plt.savefig('../Data/DG_ReducedModel/ReducedFullData/%d_%.3f.png' % (trial,je), fmt='PNG',dpi=150)
    # plt.show()
    plt.close()

def main_3():
    trial=0
    FPs=np.ones((451,5,2))*np.nan
    #### load fix point ####
    for je in np.linspace(0,0.45,451):
        FP=np.load('../Data/DG_ReducedModel/ReducedFullData/%d_%.3f.npz' % (trial,je))['FP']
        #### If even, there are two around dignal, average them ####
        if FP.shape[0]%2==0:
            FProw=FP.shape[0]
            NewFP=np.append(FP[:int(FProw/2)-1,:],(FP[int(FProw/2)-1,:]+FP[int(FProw/2),:]).reshape((1,2))/2,axis=0)
            NewFP=np.append(NewFP,FP[(int(FProw/2)+1):,:],axis=0)
            FP=NewFP
        print(trial,FP.shape)
        print(FP)
        if trial==0:
            FPs[trial,0,:]=FP[0,:]
            FPs[trial,2,:]=FP[1,:]
            FPs[trial,4,:]=FP[2,:]
            trial+=1
            continue
        #### three points, assigned to corresponding lindes ####
        if FP.shape[0]==3:
            sum1=np.sum((FPs[trial-1,0,:]-FP[0,:])**2)
            sum2=np.sum((FPs[trial-1,0,:]-FP[1,:])**2)
            sum3=np.sum((FPs[trial-1,0,:]-FP[2,:])**2)
            if sum1<sum2 and sum1<sum3:
                FPs[trial,0,:]=FP[0,:]
            if sum2<sum1 and sum2<sum3:
                FPs[trial,0,:]=FP[1,:]
            if sum3<sum1 and sum3<sum2:
                FPs[trial,0,:]=FP[2,:]

            sum1=np.sum((FPs[trial-1,2,:]-FP[0,:])**2)
            sum2=np.sum((FPs[trial-1,2,:]-FP[1,:])**2)
            sum3=np.sum((FPs[trial-1,2,:]-FP[2,:])**2)
            if sum1<sum2 and sum1<sum3:
                FPs[trial,2,:]=FP[0,:]
            if sum2<sum1 and sum2<sum3:
                FPs[trial,2,:]=FP[1,:]
            if sum3<sum1 and sum3<sum2:
                FPs[trial,2,:]=FP[2,:]

            sum1=np.sum((FPs[trial-1,4,:]-FP[0,:])**2)
            sum2=np.sum((FPs[trial-1,4,:]-FP[1,:])**2)
            sum3=np.sum((FPs[trial-1,4,:]-FP[2,:])**2)
            if sum1<sum2 and sum1<sum3:
                FPs[trial,4,:]=FP[0,:]
            if sum2<sum1 and sum2<sum3:
                FPs[trial,4,:]=FP[1,:]
            if sum3<sum1 and sum3<sum2:
                FPs[trial,4,:]=FP[2,:]

        #### five points, assigned to corresponding lindes ####
        if FP.shape[0]==5 and np.isnan(FPs[trial-1,1,0]):
            FPs[trial,0,:]=FP[0,:]
            FPs[trial,1,:]=FP[1,:]
            FPs[trial,2,:]=FP[2,:]
            FPs[trial,3,:]=FP[3,:]
            FPs[trial,4,:]=FP[4,:]
            trial+=1
            continue

        if FP.shape[0]==5:
            sum1=np.sum((FPs[trial-1,0,:]-FP[0,:])**2)
            sum2=np.sum((FPs[trial-1,0,:]-FP[1,:])**2)
            sum3=np.sum((FPs[trial-1,0,:]-FP[2,:])**2)
            sum4=np.sum((FPs[trial-1,0,:]-FP[3,:])**2)
            sum5=np.sum((FPs[trial-1,0,:]-FP[4,:])**2)
            if sum1==np.min([sum1,sum2,sum3,sum4,sum5]):
                FPs[trial,0,:]=FP[0,:]
            if sum2==np.min([sum1,sum2,sum3,sum4,sum5]):
                FPs[trial,0,:]=FP[1,:]
            if sum3==np.min([sum1,sum2,sum3,sum4,sum5]):
                FPs[trial,0,:]=FP[2,:]
            if sum4==np.min([sum1,sum2,sum3,sum4,sum5]):
                FPs[trial,0,:]=FP[3,:]
            if sum5==np.min([sum1,sum2,sum3,sum4,sum5]):
                FPs[trial,0,:]=FP[4,:]

            sum1=np.sum((FPs[trial-1,1,:]-FP[0,:])**2)
            sum2=np.sum((FPs[trial-1,1,:]-FP[1,:])**2)
            sum3=np.sum((FPs[trial-1,1,:]-FP[2,:])**2)
            sum4=np.sum((FPs[trial-1,1,:]-FP[3,:])**2)
            sum5=np.sum((FPs[trial-1,1,:]-FP[4,:])**2)
            if sum1==np.min([sum1,sum2,sum3,sum4,sum5]):
                FPs[trial,1,:]=FP[0,:]
            if sum2==np.min([sum1,sum2,sum3,sum4,sum5]):
                FPs[trial,1,:]=FP[1,:]
            if sum3==np.min([sum1,sum2,sum3,sum4,sum5]):
                FPs[trial,1,:]=FP[2,:]
            if sum4==np.min([sum1,sum2,sum3,sum4,sum5]):
                FPs[trial,1,:]=FP[3,:]
            if sum5==np.min([sum1,sum2,sum3,sum4,sum5]):
                FPs[trial,1,:]=FP[4,:]

            sum1=np.sum((FPs[trial-1,2,:]-FP[0,:])**2)
            sum2=np.sum((FPs[trial-1,2,:]-FP[1,:])**2)
            sum3=np.sum((FPs[trial-1,2,:]-FP[2,:])**2)
            sum4=np.sum((FPs[trial-1,2,:]-FP[3,:])**2)
            sum5=np.sum((FPs[trial-1,2,:]-FP[4,:])**2)
            if sum1==np.min([sum1,sum2,sum3,sum4,sum5]):
                FPs[trial,2,:]=FP[0,:]
            if sum2==np.min([sum1,sum2,sum3,sum4,sum5]):
                FPs[trial,2,:]=FP[1,:]
            if sum3==np.min([sum1,sum2,sum3,sum4,sum5]):
                FPs[trial,2,:]=FP[2,:]
            if sum4==np.min([sum1,sum2,sum3,sum4,sum5]):
                FPs[trial,2,:]=FP[3,:]
            if sum5==np.min([sum1,sum2,sum3,sum4,sum5]):
                FPs[trial,2,:]=FP[4,:]

            sum1=np.sum((FPs[trial-1,3,:]-FP[0,:])**2)
            sum2=np.sum((FPs[trial-1,3,:]-FP[1,:])**2)
            sum3=np.sum((FPs[trial-1,3,:]-FP[2,:])**2)
            sum4=np.sum((FPs[trial-1,3,:]-FP[3,:])**2)
            sum5=np.sum((FPs[trial-1,3,:]-FP[4,:])**2)
            if sum1==np.min([sum1,sum2,sum3,sum4,sum5]):
                FPs[trial,3,:]=FP[0,:]
            if sum2==np.min([sum1,sum2,sum3,sum4,sum5]):
                FPs[trial,3,:]=FP[1,:]
            if sum3==np.min([sum1,sum2,sum3,sum4,sum5]):
                FPs[trial,3,:]=FP[2,:]
            if sum4==np.min([sum1,sum2,sum3,sum4,sum5]):
                FPs[trial,3,:]=FP[3,:]
            if sum5==np.min([sum1,sum2,sum3,sum4,sum5]):
                FPs[trial,3,:]=FP[4,:]

            sum1=np.sum((FPs[trial-1,4,:]-FP[0,:])**2)
            sum2=np.sum((FPs[trial-1,4,:]-FP[1,:])**2)
            sum3=np.sum((FPs[trial-1,4,:]-FP[2,:])**2)
            sum4=np.sum((FPs[trial-1,4,:]-FP[3,:])**2)
            sum5=np.sum((FPs[trial-1,4,:]-FP[4,:])**2)
            if sum1==np.min([sum1,sum2,sum3,sum4,sum5]):
                FPs[trial,4,:]=FP[0,:]
            if sum2==np.min([sum1,sum2,sum3,sum4,sum5]):
                FPs[trial,4,:]=FP[1,:]
            if sum3==np.min([sum1,sum2,sum3,sum4,sum5]):
                FPs[trial,4,:]=FP[2,:]
            if sum4==np.min([sum1,sum2,sum3,sum4,sum5]):
                FPs[trial,4,:]=FP[3,:]
            if sum5==np.min([sum1,sum2,sum3,sum4,sum5]):
                FPs[trial,4,:]=FP[4,:]

        trial+=1
    #### exclude the false points last####
    CrossIdx1=np.where((FPs[:,0,1]-FPs[:,1,1])<2e-3)[0][0]
    CrossIdx2=np.where((FPs[:,3,1]-FPs[:,4,1])<2e-2)[0][0]
    CrossIdx=np.min([CrossIdx1,CrossIdx2])
    FPs[CrossIdx:,0,1]=np.nan
    FPs[CrossIdx:,1,1]=np.nan
    FPs[CrossIdx:,3,1]=np.nan
    FPs[CrossIdx:,4,1]=np.nan
    #### unstable and stabel states dividing the midline ####
    XIdx1=np.where((np.abs(FPs[:,1,1]-FPs[:,2,1]))==np.nanmin(np.abs(FPs[:,1,1]-FPs[:,2,1])))[0][0]
    XIdx2=np.where((np.abs(FPs[:,3,1]-FPs[:,2,1]))==np.nanmin(np.abs(FPs[:,3,1]-FPs[:,2,1])))[0][0]
    XIdx=np.min([XIdx1,XIdx2])

    #### delete nan data ####
    EffFPs1Idx=~np.isnan(FPs[:,0,1])
    Line1=np.append(np.linspace(0,0.45,451)[EffFPs1Idx].reshape((np.sum(EffFPs1Idx),1)),FPs[EffFPs1Idx,0,1].reshape((np.sum(EffFPs1Idx),1)),axis=1)

    EffFPs2Idx=~np.isnan(FPs[:,1,1])
    Line2=np.append(np.linspace(0,0.45,451)[EffFPs2Idx].reshape((np.sum(EffFPs2Idx),1)),FPs[EffFPs2Idx,1,1].reshape((np.sum(EffFPs2Idx),1)),axis=1)

    EffFPs3Idx=~np.isnan(FPs[:,2,1])*(np.arange(FPs[:,2,1].size)<XIdx)
    Line3=np.append(np.linspace(0,0.45,451)[EffFPs3Idx].reshape((np.sum(EffFPs3Idx),1)),FPs[EffFPs3Idx,2,1].reshape((np.sum(EffFPs3Idx),1)),axis=1)

    EffFPs4Idx=~np.isnan(FPs[:,2,1])*(np.arange(FPs[:,2,1].size)>=XIdx)
    Line4=np.append(np.linspace(0,0.45,451)[EffFPs4Idx].reshape((np.sum(EffFPs4Idx),1)),FPs[EffFPs4Idx,2,1].reshape((np.sum(EffFPs4Idx),1)),axis=1)

    EffFPs5Idx=~np.isnan(FPs[:,3,1])
    Line5=np.append(np.linspace(0,0.45,451)[EffFPs5Idx].reshape((np.sum(EffFPs5Idx),1)),FPs[EffFPs5Idx,3,1].reshape((np.sum(EffFPs5Idx),1)),axis=1)

    EffFPs6Idx=~np.isnan(FPs[:,4,1])
    Line6=np.append(np.linspace(0,0.45,451)[EffFPs6Idx].reshape((np.sum(EffFPs6Idx),1)),FPs[EffFPs6Idx,4,1].reshape((np.sum(EffFPs6Idx),1)),axis=1)
    print(Line1.shape)

    #### smooth the line ####
    NewLine1=(Line1[1:,:]+Line1[:-1,:])/2
    NewLine1=np.append([[Line1[0,0],Line1[0,1]]],NewLine1,axis=0)
    NewLine1=np.append(NewLine1,[[Line1[-1,0],Line1[-1,1]]],axis=0)
    for i in range(50):
        NewLine1=(NewLine1[1:,:]+NewLine1[:-1,:])/2
        NewLine1=np.append([[Line1[0,0],Line1[0,1]]],NewLine1,axis=0)
        NewLine1=np.append(NewLine1,[[Line1[-1,0],Line1[-1,1]]],axis=0)
    Line1=NewLine1

    NewLine2=(Line2[1:,:]+Line2[:-1,:])/2
    NewLine2=np.append([[Line2[0,0],Line2[0,1]]],NewLine2,axis=0)
    NewLine2=np.append(NewLine2,[[Line2[-1,0],Line2[-1,1]]],axis=0)
    for i in range(50):
        NewLine2=(NewLine2[1:,:]+NewLine2[:-1,:])/2
        NewLine2=np.append([[Line2[0,0],Line2[0,1]]],NewLine2,axis=0)
        NewLine2=np.append(NewLine2,[[Line2[-1,0],Line2[-1,1]]],axis=0)
    Line2=NewLine2

    NewLine3=(Line3[1:,:]+Line3[:-1,:])/2
    NewLine3=np.append([[Line3[0,0],Line3[0,1]]],NewLine3,axis=0)
    NewLine3=np.append(NewLine3,[[Line3[-1,0],Line3[-1,1]]],axis=0)
    for i in range(50):
        NewLine3=(NewLine3[1:,:]+NewLine3[:-1,:])/2
        NewLine3=np.append([[Line3[0,0],Line3[0,1]]],NewLine3,axis=0)
        NewLine3=np.append(NewLine3,[[Line3[-1,0],Line3[-1,1]]],axis=0)
    Line3=NewLine3

    NewLine4=(Line4[1:,:]+Line4[:-1,:])/2
    NewLine4=np.append([[Line4[0,0],Line4[0,1]]],NewLine4,axis=0)
    NewLine4=np.append(NewLine4,[[Line4[-1,0],Line4[-1,1]]],axis=0)
    for i in range(50):
        NewLine4=(NewLine4[1:,:]+NewLine4[:-1,:])/2
        NewLine4=np.append([[Line4[0,0],Line4[0,1]]],NewLine4,axis=0)
        NewLine4=np.append(NewLine4,[[Line4[-1,0],Line4[-1,1]]],axis=0)
    Line4=NewLine4

    NewLine5=(Line5[1:,:]+Line5[:-1,:])/2
    NewLine5=np.append([[Line5[0,0],Line5[0,1]]],NewLine5,axis=0)
    NewLine5=np.append(NewLine5,[[Line5[-1,0],Line5[-1,1]]],axis=0)
    for i in range(50):
        NewLine5=(NewLine5[1:,:]+NewLine5[:-1,:])/2
        NewLine5=np.append([[Line5[0,0],Line5[0,1]]],NewLine5,axis=0)
        NewLine5=np.append(NewLine5,[[Line5[-1,0],Line5[-1,1]]],axis=0)
    Line5=NewLine5

    NewLine6=(Line6[1:,:]+Line6[:-1,:])/2
    NewLine6=np.append([[Line6[0,0],Line6[0,1]]],NewLine6,axis=0)
    NewLine6=np.append(NewLine6,[[Line6[-1,0],Line6[-1,1]]],axis=0)
    for i in range(50):
        NewLine6=(NewLine6[1:,:]+NewLine6[:-1,:])/2
        NewLine6=np.append([[Line6[0,0],Line6[0,1]]],NewLine6,axis=0)
        NewLine6=np.append(NewLine6,[[Line6[-1,0],Line6[-1,1]]],axis=0)
    Line6=NewLine6
    
    #### conpensate points ####
    meannlast12=(3*Line1[-1,1]+Line2[-1,1])/4
    Line1=np.append(Line1,[[0.293,meannlast12]],axis=0)
    Line2=np.append(Line2,[[0.293,meannlast12]],axis=0)
    Line2=np.append([[Line3[-1,0],Line3[-1,1]]],Line2,axis=0)
    Line4=np.append([[Line3[-1,0],Line3[-1,1]]],Line4,axis=0)
    Line5=np.append([[Line3[-1,0],Line3[-1,1]]],Line5,axis=0)
    meannlast56=(Line5[-1,1]+2*Line6[-1,1])/3
    Line5=np.append(Line5,[[0.293,meannlast56]],axis=0)
    Line6=np.append(Line6,[[0.293,meannlast56]],axis=0)

    np.savez_compressed('../Data/DG_ReducedModel/bifurcation.npz',L1=Line1,L2=Line2,L3=Line3,L4=Line4,L5=Line5,L6=Line6)

    fig2,ax2=plt.subplots()
    ax2.plot(Line1[:,0],Line1[:,1],linewidth=2)
    ax2.plot(Line2[:,0],Line2[:,1],linestyle=':',linewidth=2)
    ax2.plot(Line3[:,0],Line3[:,1],linestyle=':',linewidth=2)
    ax2.plot(Line4[:,0],Line4[:,1],linewidth=2)
    ax2.plot(Line5[:,0],Line5[:,1],linestyle=':',linewidth=2)
    ax2.plot(Line6[:,0],Line6[:,1],linewidth=2)
    plt.savefig('../Data/DG_ReducedModel/bifurcation.png',fmt='PNG',dpi=300)
    plt.show()

if __name__ == '__main__':
    main_0()
    # trial=1000
    # for je in np.linspace(0.16,0.36,10):
    #     for ji in np.linspace(0.04,0.4,10):
    #         for i0 in np.linspace(0.42,0.62,10):
    #             print(trial)
    #             main_1(trial,je,ji,i0)
    #             trial+=1

    # trial=0
    # for je in np.linspace(0,0.45,451):
    #     print(trial)
    #     main_2(trial,je)
    #     trial+=1

    # main_3()
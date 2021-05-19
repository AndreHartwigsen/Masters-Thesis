import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.io import fits
import pandas as pd
import os
plt.close('all')


def Peak(w,T):
    if w[0]>w[-1]:
        w = w[::-1] ; T = T[::-1]
    a = np.trapz(w*T,w)
    b = np.trapz(T,w)
    return a/b
def FWHM(w,T):
    if w[0]>w[-1]:
        w = w[::-1] ; T = T[::-1]
    HM = 0.5*np.max(T)
    larger = np.where(T >= HM)[0]
    x1 = np.linspace(w[larger[0]-2],w[larger[0]+2],500)
    x2 = np.linspace(w[larger[-1]-2],w[larger[-1]+2],500)
    T1 = np.interp(x1,w,T)
    T2 = np.interp(x2,w,T)
    x1 = x1[np.where(T1 >= HM)[0][0]]
    x2 = x2[np.where(T2 <= HM)[0][0]]
    return x2-x1


def swapPositions(list, pos1, pos2): 
    list[pos1], list[pos2] = list[pos2], list[pos1] 
    return list
def wmean(x,w):
    m = np.sum(x*w)/np.sum(w)
    return m

files = []
cen = []
for name in os.listdir():
    if name.endswith('.txt'):
        files.append(name)
        cen.append(wmean(np.loadtxt(name)[:,0],np.loadtxt(name)[:,1]))
cens = np.sort(cen)
com,t1,t2 = np.intersect1d(cen,cens,return_indices=True)

files2 = []
for i in range(len(t1)):
    files2.append(files[t1[i]])
files = swapPositions(files2,5,9)
files = swapPositions(files,32,33)
files = swapPositions(files,18,21)
files = swapPositions(files,18,20)


fig1 = plt.figure(figsize=(8,2.3),dpi=200)
ax = plt.subplot(111)
vista = (0.5,0.1,0.1)
IRAC = (0.1,0.5,0.1)
SUBARU = (0.1,0.1,0.5)
HSC = (0.5,0.5,0.1)
HERS = (0.1,0.5,0.5)
ACS = (0.1,0.2,0.3)

LW = 0.5

alp = 0.3
wavd = 10000
for i in range(len(files)):
    if 'SUNARROW' in files[i]:
        data = np.loadtxt(files[i])
        cut = np.where(data[:,1] > 0.001)[0]
        plt.semilogx(data[:,0][cut]/wavd,0.3*data[:,1][cut],'-k',alpha=alp,lw=LW,zorder=90)#,label=files[i][:files[i].index('_')])
        ax.fill_between(data[:,0][cut]/wavd,0.3*data[:,1][cut],np.zeros(len(cut)),color='black',alpha=0.4,zorder=90)
        alp += 0.04

excl = 'SPIREPACSMIPS'

for i in range(len(files)):
    if files[i][:4] not in excl:
        col = (0,0,0)
        data = np.loadtxt(files[i])
        if 'VISTA' in files[i]:
            col = vista
            vista = (vista[0]+0.07,0.1,0.1)
        if 'SPITZER' in files[i]:
            col = IRAC
            IRAC = (0.1,0.07+IRAC[1],0.1)
        if 'SUBARU' in files[i]:
            col = SUBARU
            SUBARU = (0.1,0.1,0.07+SUBARU[2])
        if 'HSC' in files[i]:
            col = HSC
            HSC = (HSC[0]+0.07,HSC[1]+0.07,0.1)
        if 'HERSHEL' in files[i]:
            col = HERS
            HERS = (0.1,HERS[1]+0.1,HERS[2]+0.07)
        if 'ACS' in files[i]:
            col = ACS
        if 'NARROW' in files[i] and 'SUNARROW' not in files[i]:
            mod = ['--',0.3]
        else:
            mod = ['',1]
        if 'SUNARROW' not in files[i]:
            cut = np.where(data[:,1] > 0.001)[0]
            plt.semilogx(data[:,0][cut]/wavd,mod[1]*data[:,1][cut],mod[0],color=col,lw=LW,label=files[i][:files[i].index('_')])
            ax.fill_between(data[:,0][cut]/wavd,mod[1]*data[:,1][cut],np.zeros(len(cut)),color=col,alpha=0.3)

plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left',ncol=2)#,loc=1)
plt.xlabel('Wavelength [$\mu$m]')
plt.xlim(left=1300/wavd,right=1e+5/wavd)
plt.ylabel('Transmission')
ticks = [0.2,0.3,0.5,0.7,1,2,3,4,5,7,10]
labels = ['%.2g'%i for i in ticks]
plt.xticks(ticks,labels)
fig1.subplots_adjust(top=0.995,bottom=0.196,left=0.065,right=0.716,hspace=0.135,wspace=0.2)
# plt.tight_layout()

peak = []
fwhm = []
name = []

for i in range(len(files)):
    data = np.loadtxt(files[i])
    peak.append(Peak(data[:,0],data[:,1]))
    fwhm.append(FWHM(data[:,0],data[:,1]))
    name.append(files[i][:files[i].index('_')])
    #print("%s with mean %.3g and FWHM %.3g" % (name[-1],peak[-1],fwhm[-1]) )
#frame = {'Filter':name , 'Mean':peak , 'FWHM':fwhm}
#pd.DataFrame(frame).to_csv('ManualFilters.csv',sep=',',index=False)




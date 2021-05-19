import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import scipy
from scipy.optimize import curve_fit

plt.close('all')


def voight(X,mu,s,g,N,a):
    x  = X - mu
    return scipy.special.voigt_profile(x,s,g)*N+a
def wmean(x,sx,axis=None):#Weighted mean
    w = 1/sx**2
    m = np.sum(x*w,axis=axis)/np.sum(w,axis=axis)
    sm = 1/(np.sqrt(np.sum(w,axis=axis)))
    return m,sm


w = [3518.497314453125]
for i in range(1149):
    w.append(w[-1]+3.402486562728882)
w = np.array(w)

file = fits.open('flux_obj.ms_1d.fits')
fitsdata = file[0].data[:,0]
F = fitsdata[0]
Fr = fitsdata[3]


def fit(mu=7254):#Fit voight profile, return wavelength range of fit, fit, fit coefficients and uncertainty of coefficients
    guess = (mu,10,0,2e-14,1e-17)
    s1 = (w>guess[0]-40) & (w<guess[0]+40) 
    ws = w[s1]
    Fs = F[s1]
    Frs = Fr[s1]
    ab,cov = curve_fit(voight,ws,Fs,sigma=Frs,p0=guess,maxfev=5000)
    wl = np.linspace(min(ws),max(ws),500)
    return wl,voight(wl,*ab),ab,np.sqrt(np.diag(cov))

#             H-gamma   H-beta  OIII   OIII   H-alpha
Llabel = [r'H$\gamma$',r'H$\beta$','[OIII]','[OIII]',r'H$\alpha$']
L0 = np.array([4341.684,4862.683,4960.295,5008.240,6564.614]) #Vaccum wavelengths

cen = np.array([4781,5357,5463,5516,7254]) #Estimated peaks



AB = []
ABr = []
plt.figure(figsize=(6,2.5),dpi=150)
plt.errorbar(w/10000,F,Fr,fmt='.-k',ms=2)
for i,mu in enumerate(cen):
    w1,f1, ab, sig = fit(mu)
    plt.plot(w1/10000,f1,'r')
    plt.annotate(Llabel[i], xy=(ab[0]/10000,np.max(f1)),ha='center')
    AB.append(ab[0])
    ABr.append(sig[0])
AB = np.asarray(AB)
ABr = np.asarray(ABr)
plt.xlabel(r'$\lambda$ [$\mu$m]')
plt.ylabel(r'Flux [erg/s/cm$^2$/Å]')
#plt.gca().set_xscale('log')
ticks = np.array([0.4,0.5,0.6,0.7,0.8,0.9,1])*10000
labels = ['%.10g'%i for i in ticks]
#plt.xticks(ticks,labels)
# plt.gca().invert_yaxis()
#plt.ylim([-1e-16,np.max(F1)*1.1])
plt.ylim([1e-18,9e-16])
plt.subplots_adjust(top=0.936,bottom=0.181,left=0.071,right=0.998,hspace=0.2,wspace=0.2)
plt.figure(figsize=(3,2.5),dpi=150)
plt.errorbar(w/10000,F,Fr,fmt='.-k',ms=2)
for i,mu in enumerate(cen):
    w1,f1, ab, sig = fit(mu)
    plt.plot(w1/10000,f1,'r')
    plt.annotate(Llabel[i], xy=(ab[0]/10000,np.max(f1)),ha='center')

plt.xlim([5300/10000,5550/10000])
plt.ylim([1e-18,9e-16])
plt.ylabel('')
plt.yticks([])
plt.subplots_adjust(top=0.936,bottom=0.181,left=0.061,right=0.932,hspace=0.2,wspace=0.2)
plt.xlabel(r'$\lambda$ [$\mu$m]')

z = AB/L0-1
zr = ABr/L0

Z,Zr = wmean(z,zr)


print(Z,Zr)
















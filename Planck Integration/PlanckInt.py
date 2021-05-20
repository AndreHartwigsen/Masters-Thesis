import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.io import fits
import pandas as pd
from scipy.optimize import curve_fit
plt.close('all')

# c = np.loadtxt('coeff.txt')
# ca = np.zeros([6,48])
# for i in range(6):
#     ca[i] = c[i*48:i*48+48]
# ca = ca[:,:45]

def k(lam): 
    # http://articles.adsabs.harvard.edu/pdf/1990ApJS...72..163F
    x = (1/1000000)/lam
    x0 = 4.608
    y = 0.994
    c = [-0.687,0.891,2.550,0.504]
    D = x**2/( (x**2-x0**2)**2 + x**2*y**2 )
    F = 0.5392*(x - 5.9)**2 + 0.05644*(x - 5.9)**3
    k = c[0] + c[1]*x + c[2]*D + c[3]*F
    return 10**k

x = np.linspace(1e+3,1e+5,5000)*1e-10

fig1 = plt.figure(figsize=(7,2),dpi=220)
fig1.subplots_adjust(top=0.995,bottom=0.22,left=0.087,right=0.999,hspace=0.2,wspace=0.2)
ax = plt.subplot(111)
#plt.plot(1/(x*1e+6),np.log10(k(x)),'-k')
plt.plot(x*1e+5,k(x),'-k')
plt.gca().set_xscale('log')
plt.gca().set_yscale('log')
plt.xlabel(r'$\lambda$ [$\mu$m]')
plt.ylabel('$k(\lambda-V)$')




#%%

x = np.linspace(500,500000,5000)*1e-10

def planck(x,T=5700):
    a = 1.191042953e-16
    b = 0.01438777354/T
    return a/x**5 * 1/(np.exp(b/x)-1)



zwave = np.loadtxt('z.txt')[:,0]*1e-10
zTrans = np.loadtxt('z.txt')[:,1]
Kswave = np.loadtxt('Ks.txt')[:,0]*1e-10
KsTrans = np.loadtxt('Ks.txt')[:,1]
Bwave = np.loadtxt('B.txt')[:,0]*1e-10
BTrans = np.loadtxt('B.txt')[:,1]
Vwave = np.loadtxt('V.dat')[:,0]*1e-10
VTrans = np.loadtxt('V.dat')[:,1]
NBwave = np.loadtxt('NB118.dat')[:,0]*1e-10
NBTrans = np.loadtxt('NB118.dat')[:,1]

Hwave = np.loadtxt('H.txt')[:,0]*1e-10
HTrans = np.loadtxt('H.txt')[:,1]
Jwave = np.loadtxt('J.txt')[:,0]*1e-10
JTrans = np.loadtxt('J.txt')[:,1]
Ywave = np.loadtxt('Y.txt')[:,0]*1e-10
YTrans = np.loadtxt('Y.txt')[:,1]
wz = np.trapz(zTrans,zwave)
wKs = np.trapz(KsTrans,Kswave)
wB = np.trapz(BTrans,Bwave)
wH = np.trapz(HTrans,Hwave)
wJ = np.trapz(JTrans,Jwave)
wY = np.trapz(YTrans,Ywave)
def zf(x,filt='z'):
    if filt == 'z':
        return np.interp(x,zwave,zTrans,left=0,right=0)
    if filt == 'B':
        return np.interp(x,Bwave,BTrans,left=0,right=0)
    if filt == 'V':
        return np.interp(x,Vwave,VTrans,left=0,right=0)
    if filt == 'Y':
        return np.interp(x,Ywave,YTrans,left=0,right=0)
    if filt == 'J':
        return np.interp(x,Jwave,JTrans,left=0,right=0)
    if filt == 'H':
        return np.interp(x,Hwave,HTrans,left=0,right=0)
    if filt == 'Ks':
        return np.interp(x,Kswave,KsTrans,left=0,right=0)
    if filt == 'NB118':
        return np.interp(x,NBwave,NBTrans,left=0,right=0)



def inte(x,T=5700,filt='none'):
    if filt in ['z','B','V','H','J','H','Ks','Y','NB118']:
        return np.trapz(planck(x,T)*zf(x,filt),x)
    else:
        return np.trapz(planck(x,T),x)



Tp = [2000,3000, 5000, 8000]


plt.figure(figsize=(16,10),dpi=80)
plt.subplot(321)
for Ti in Tp:
    plt.loglog(x/1e-10,planck(x,Ti),label='T=%i'%(Ti))
plt.gca().set_ylim(bottom=100)
plt.legend()
plt.xlabel(r'$\lambda$ [Å]')
plt.ylabel(r'Intensity')
plt.title('Black Body spectrum')




plt.subplot(322)
for Ti in Tp:
    y = []
    for i in range(len(x)):
        y.append(inte(x[:i],Ti))
    plt.loglog(x/1e-10,y,label='T=%i'%(Ti))
plt.legend()
plt.gca().set_ylim(bottom=10)
plt.xlabel(r'$\lambda$ [Å]')
plt.ylabel('Accumulative intensity')
plt.title('Integral of spectrum')


x = np.linspace(9000,30000,5000)*1e-10
dx = x[1]-x[0]

plt.subplot(323)
for Ti in Tp:
    plt.plot(x/1e-10,planck(x,Ti)*zf(x,'H'),label='T=%i'%(Ti))
plt.xlabel(r'$\lambda$ [Å]')
plt.xlim([1e+10*min(Hwave),1e+10*max(Hwave)])
plt.title(r'Convulution of $H$ filter')
#plt.ylim([0,3.5e+12])

plt.subplot(324)
for Ti in Tp:
    plt.plot(x/1e-10,planck(x,Ti)*zf(x,'Ks'),label='T=%i'%(Ti))
plt.legend()
plt.xlabel(r'$\lambda$ [Å]')
plt.xlim([1e+10*min(Kswave),1e+10*max(Kswave)])
plt.title(r'Convulution of $K_s$ filter')
#plt.ylim([0,3.5e+12])



plt.subplot(325)
for Ti in Tp:
    y = []
    for i in range(len(x)):
        y.append(inte(x[:i],Ti,filt='H'))
    plt.loglog(x/1e-10,y,label='T=%i'%(Ti))
plt.legend()
plt.xlabel(r'$\lambda$ [Å]')
plt.title(r'Integral for convolution of $H$ filter')
plt.gca().set_ylim(bottom=10)


plt.subplot(326)
for Ti in Tp:
    y = []
    for i in range(len(x)):
        y.append(inte(x[:i],Ti,filt='Ks'))
    plt.loglog(x/1e-10,y,label='T=%i'%(Ti))
plt.legend()
plt.xlabel(r'$\lambda$ [Å]')
plt.title(r'Integral for convolution of $K_s$ filter')
plt.gca().set_ylim(bottom=10)


plt.tight_layout()
plt.savefig('curves.png',dpi=300)



#%%

def conv(x,T=5700,filt='z',noPlanck = False):
    if noPlanck:
        p = 1
    else:
        p = planck(x,T)
    if filt == 'z':
        a = np.interp(x,zwave,zTrans,left=0,right=0)
    if filt == 'B':
        a = np.interp(x,Bwave,BTrans,left=0,right=0)
    if filt == 'V':
        a = np.interp(x,Vwave,VTrans,left=0,right=0)
    if filt == 'H':
        a = np.interp(x,Hwave,HTrans,left=0,right=0)
    if filt == 'J':
        a = np.interp(x,Jwave,JTrans,left=0,right=0)
    if filt == 'Ks':
        a = np.interp(x,Kswave,KsTrans,left=0,right=0)
    if filt == 'NB118':
        a = np.interp(x,NBwave,NBTrans,left=0,right=0)
    if filt == 'Y':
        a = np.interp(x,Ywave,YTrans,left=0,right=0)
    a *= p
    i = np.where(a>0)[0]
    return x[i]/1e-10,a[i]



x2 = np.linspace(1000,50000,5000)*1e-10
sel = ['z','B','V','H','Y','J','Ks','NB118']

plt.figure(figsize=(8,6),dpi=150)

plt.plot(x2/1e-10,planck(x2,3000),'k',label='Blackbody spectrum')

for s in sel:
    plt.plot(*conv(x2,3000,s),label=s)
plt.legend()
plt.xlabel(r'$\lambda$ [Å]')
plt.xlim([1000,25000])
plt.ylabel(r'Relative intensity [W m$^{-2}$ m$^{-1}$]')
plt.tight_layout()



sel = ['H','Y','J','Ks','NB118','z','B','V']
lab = ['H','Y','J','K_s','NB118','z','B','V']
fig1 = plt.figure(figsize=(8,3.5),dpi=150)
fig1.subplots_adjust(top=0.998,bottom=0.135,left=0.066,right=0.998,hspace=0.2,wspace=0.2)
plt.plot(np.array([.3300,2.7000]),[0,0],'--k')
i = 0
for s in sel:
    xx,aa = conv(x2,3000,s,True)
    plt.plot(xx/10000,aa,label="$%s$"%lab[i])
    i += 1
plt.legend()
plt.xlabel(r'$\lambda$ [$\mu$m]')
plt.ylabel('Transmission')
plt.xlim([.3300,2.7000])
plt.savefig('filtercurves.eps')




#%%

zloga = []
yloga = []
xloga = []
Temp = np.linspace(2000,7000,500)
for T in Temp:
     #xloga.append(np.log10( inte(x,T,'Y') / wY ) -np.log10( inte(x,T,'J' ) / wJ ))
     #yloga.append(np.log10( inte(x,T,'H') / wH ) -np.log10( inte(x,T,'Ks') / wKs))
     xloga.append( np.log10(inte(x,T,'Y') / inte(x,T,'J' ) * wJ /wY) )
     yloga.append( np.log10(inte(x,T,'H') / inte(x,T,'Ks') * wKs/wH) )
     zloga.append( np.log10(inte(x,T,'J') / inte(x,T,'Ks') * wKs/wY) )
zloga = -np.array(zloga)
yloga = -np.array(yloga)
xloga = -np.array(xloga)







plt.figure(figsize=(20,8),dpi=100)
plt.subplot(131)
plt.plot(Temp,xloga)
plt.xlabel('T [K]')
plt.ylabel(r'$Y-J$')
plt.title(r'$Y-J$ magnitude')
plt.tight_layout()


plt.subplot(132)
plt.plot(Temp,yloga)
plt.xlabel('T [K]')
plt.ylabel(r'$H-K_s$')
plt.title(r'$H-K_s$ magnitude')
plt.tight_layout()

plt.subplot(133)
plt.plot(Temp,zloga)
plt.xlabel('T [K]')
plt.ylabel(r'$J-K_s$')
plt.title(r'$J-K_s$ magnitude')
plt.tight_layout()

plt.savefig('H-Ks at temp.png',dpi=300)



#%%

def cum(x): #Cumulative function
    y = np.zeros(len(x))
    for i in range(len(x)):
        y[i] = np.sum(x[:i])
    return y/y[-1]

def gen(M,CDF,N): 
    rand = np.random.rand(N)
    return np.interp(rand,CDF,M)



M = 10 ** (np.linspace(-3,1.8,1000))
dM = M[1:]-M[:-1]
M = M[:-1]

def N(M,dM):
    # https://iopscience.iop.org/article/10.1086/376392/pdf
    M = np.log10(M)
    A = np.ones(len(M))*4.43e-2
    alp = np.ones(len(M))*1.3
    
    sel4 = np.where(M <= 0)
    A[sel4] = 0.158
    mc = 0.079
    sig = 0.69
    xi = A*(10**M)**(-alp)
    xi[sel4] = A[sel4] * np.exp( -( M[sel4]-np.log10(mc) )**2 / (2*sig**2) )
    return xi

def Radius(M):
    R = 0.1374 + 0.5419*np.log10(M)
    sel = np.where(M < 1.66)
    R[sel] = 0.0225 + 0.9349*np.log10(M[sel])
    return 10**R
def Lumin(M):
    L = 32000*M
    sel1 = np.where( (2.00 < M) & (M < 55) )[0]
    sel2 = np.where( (0.43 < M) & (M <= 2) )[0]
    sel3 = np.where( M <= 0.43)[0]
    L[sel1] = 1.4*(M[sel1])**3.5
    L[sel2] = 1.0*(M[sel2])**4.0
    L[sel3] = .23*(M[sel3])**2.3
    
    sela = np.where( (0.2 < M) & (M < 0.85) )[0]
    a = -141.7*M[sela]**4 + 232.4*M[sela]**3 - 129.1*M[sela]**2 + 33.29*M[sela] + 0.215
    L[sela] = M[sela]**a
    return L


def Temp(M):
    R = Radius(M)
    L = Lumin(M)
    return 4851.420661*(L/(R)**2)**(1/4)


plt.figure(figsize=(8,4),dpi=100)

plt.subplot(1,2,1)
plt.loglog(M,N(M,dM)/np.trapz(N(M,dM),M))

plt.subplot(1,2,2)
plt.semilogx(M,cum(N(M,dM)))
plt.tight_layout()





#%%

def expApproach(X,k):
    x = X*1e+10
    B = 0.5 #where should the graph be at xc [Å]
    xc = 10000
    A = (1-B)/(np.exp(-xc/k))
    return 1 - A * np.exp(-x/k)
Upper = 50000
Lower = 100
kRand = np.random.rand(10000)*(Upper-Lower) + Lower

def EllipSpec(X,z = 0):
    x = X*1e+10
    l = .1
    y = np.zeros(len(x))+l
    C1 = 3000*(1+z)
    C2 = 6000*(1+z)
    c1 = np.where( (x >= C1) & (x <= C2) )[0]
    c2 = np.where( x > C2)[0]
    y[c2] = 1
    a = (l-1) / (C1-C2)
    b = (C1*1-C2*l) / (C1-C2)
    y[c1] = a*x[c1]+b
    return y+np.random.rand(len(x))*1e-4
    
Upper = 9
Lower = 1
zRand = np.random.rand(10000)*(Upper-Lower) + Lower


def Spiral(X,z=0):
    x = X*1e+10
    l = .1
    y = np.zeros(len(x))+l
    C1 = 3500*(1+z)
    C2 = 5000*(1+z)
    C3 = 8000*(1+z)
    c1 = np.where( (x >= C1) & (x <= C2) )[0]
    c2 = np.where( x > C2)[0]
    a = (l-1) / (C1-C2)
    b = (C1*1-C2*l) / (C1-C2)
    y[c1] = a*x[c1]+b
    
    a = (1-0.5) / (C2-C3)
    b = (C2*0.5-C3*1) / (C2-C3)
    y[c2] = a*x[c2]+b
    cf = np.where(y < l)[0]
    y[cf] = l
    return y+np.random.rand(len(x))*1e-4


Ti = 10**np.linspace(3.42,4.7,10000)
IY = np.zeros(len(Ti))
IJ = np.zeros(len(Ti))
IH = np.zeros(len(Ti))
IKs= np.zeros(len(Ti))
for i in range(len(Ti)):
    IY[i] = inte(Ywave,Ti[i],'Y')
    IJ[i] = inte(Jwave,Ti[i],'J')
    IH[i] = inte(Hwave,Ti[i],'H')
    IKs[i] = inte(Kswave,Ti[i],'Ks')

Ms = gen(M,cum(N(M,dM)),10000000)
Ts = Temp(Ms)

r = np.random.rand(len(Ts))*0.025
phi = 2*np.pi*np.random.rand(len(Ts))

xloga = -2.5*np.log10( np.interp(Ts,Ti,IY) / np.interp(Ts,Ti,IJ)  *wJ/wY  + r*np.cos(phi) )
yloga = -2.5*np.log10( np.interp(Ts,Ti,IH) / np.interp(Ts,Ti,IKs) *wKs/wH + r*np.sin(phi) )

fxloga = -2.5*np.log10( np.trapz(zf(x,'Y'),x,dx) / np.trapz(zf(x,'J'),x,dx) *wJ/wY  +r*np.cos(phi) )
fyloga = -2.5*np.log10( np.trapz(zf(x,'H'),x,dx) / np.trapz(zf(x,'Ks'),x,dx) *wKs/wH +r*np.sin(phi) )



exloga = []
eyloga = []
Exloga = []
Eyloga = []
Sxloga = []
Syloga = []
for i in range(len(kRand)):
    Spec = expApproach(x,kRand[i])
    exloga.append(-2.5*np.log10( np.trapz(zf(x,'Y')*Spec,x,dx) / np.trapz(zf(x,'J')*Spec,x,dx)  *wJ/wY + r[i]*np.cos(phi[i]) ))
    eyloga.append(-2.5*np.log10( np.trapz(zf(x,'H')*Spec,x,dx) / np.trapz(zf(x,'Ks')*Spec,x,dx) *wKs/wH + r[i]*np.sin(phi[i]) ))
    Spec = EllipSpec(x,zRand[i])
    Exloga.append(-2.5*np.log10( np.trapz(zf(x,'Y')*Spec,x,dx) / np.trapz(zf(x,'J')*Spec,x,dx)  *wJ/wY + r[i]*np.cos(phi[i]) ))
    Eyloga.append(-2.5*np.log10( np.trapz(zf(x,'H')*Spec,x,dx) / np.trapz(zf(x,'Ks')*Spec,x,dx) *wKs/wH + r[i]*np.sin(phi[i]) ))
    Spec = Spiral(x,zRand[i])
    Sxloga.append(-2.5*np.log10( np.trapz(zf(x,'Y')*Spec,x,dx) / np.trapz(zf(x,'J')*Spec,x,dx)  *wJ/wY + r[i]*np.cos(phi[i]) ))
    Syloga.append(-2.5*np.log10( np.trapz(zf(x,'H')*Spec,x,dx) / np.trapz(zf(x,'Ks')*Spec,x,dx) *wKs/wH + r[i]*np.sin(phi[i]) ))
exloga = np.array(exloga)
eyloga = np.array(eyloga)
Exloga = np.array(Exloga)
Eyloga = np.array(Eyloga)
Sxloga = np.array(Sxloga)
Syloga = np.array(Syloga)

Mg = np.copy(Ms)
#%%

plt.close(fig=12)
plt.figure(12)
plt.plot(x*1e+10,expApproach(x,7000),'g',label='Exponential')
plt.plot(x*1e+10,EllipSpec(x,3),'b',label='Linear Rise')
plt.plot(x*1e+10,Spiral(x,2),'m',label='Spiral')
plt.plot(x*1e+10,np.ones(len(x))*0.5,'--r',label='Flat')
Tpl = 6500
plt.plot(x*1e+10,planck(x,Tpl)/np.max(planck(x,Tpl)),'k',label='Planck (%iK)'%Tpl)
plt.legend()

#%%

plt.close(fig=7);plt.close(fig=8)

MassNorm = np.trapz(N(M,dM),M)

fig7 = plt.figure(7,figsize=(4,3),dpi=200)
fig7.subplots_adjust(top=0.999,bottom=0.148,left=0.166,right=0.973,hspace=0.2,wspace=0.2)
h,be = np.histogram(Mg,np.logspace(min(np.log10(M)),max(np.log10(M)),200))
bc = (be[:-1]+be[1:])/2
bw = be[1:]-be[:-1]
plt.bar(bc,h*MassNorm/(np.sum(bw*h)),width=bw,fc='k')
plt.plot(M,N(M,dM),'--r')
plt.yscale('log')
plt.xscale('log')
plt.xlabel('Mass [$M_\odot$]')
plt.ylabel(r'$\xi(m)$ [$(\log M_\odot)^{-1}$pc$^{-3}$]')
#plt.savefig('../Thesis/fig/ChabrierIMF.eps')



plt.figure(8,figsize=(10,8),dpi=100)
h,be = np.histogram(Temp(Mg),100)
bc = (be[:-1]+be[1:])/2
bw = bc[1]-bc[0]
plt.bar(bc,h/(np.sum(bw*h)),width=be[1:]-be[:-1],fc='k')
plt.yscale('log')
plt.xlabel('Temperature [K]')


lims = [[-.5,.5],[-1,.4]]
plt.close(fig=9)
plt.figure(9,figsize=(4,3),dpi=100)
plt.hist2d(Sxloga,Syloga,200,lims,norm=LogNorm(vmax=10),cmap='Reds',label='Spiral Spectrum')
plt.hist2d(xloga,yloga,200,lims,norm=LogNorm(),cmap='binary',label='Star spectrum')
plt.hist2d(Exloga,Eyloga,200,lims,norm=LogNorm(),cmap='Blues',label='Ellip Spectrum')
plt.hist2d(exloga,eyloga,200,lims,norm=LogNorm(),cmap='Greens',label='Exp Rise Spectrum')
#plt.colorbar()
labels = ['Spirals','Stars','Exp rise','Elliptical']
handles = [plt.Rectangle((0,0),1,1,color=c,ec="k") for c in ['red','black','green','blue']]
plt.legend(handles,labels,loc='upper left')
plt.xlabel(r'$Y-J$ [mag]')
plt.ylabel(r'$H-K_s$ [mag]')
plt.tight_layout()



#%%

def tt(T,a,i,b):
    return a*T**i + b 
def tte(T,a,b,c):
    return np.exp(a*T)*b + c



plt.close(fig=10)

plt.figure(10,figsize=(6,4),dpi=100)
plt.plot(Ti,IY,label='Y')
plt.plot(Ti,IH,label='H')
plt.plot(Ti,IJ,label='J')
plt.plot(Ti,IKs,label='Ks')
plt.xscale('log')
plt.xlabel('Temperature [K]')
plt.ylabel('Flux equivalent')
plt.legend()
plt.tight_layout()

ab,cov = curve_fit(tt,np.log10(Ti),IH)
abe,cov = curve_fit(tte,np.log10(Ti),IH)



plt.close(fig=11)
plt.figure(11,figsize=(6,4),dpi=100)

plt.semilogx(Ti,(IH-tt(np.log10(Ti),*ab))/IH*100,'--r',label=r'Poly $%.2g T^{%.4g} %.4g$' % (ab[0],ab[1],ab[2]))
plt.semilogx(Ti,(IH-tte(np.log10(Ti),*abe))/IH*100,'--g',label='Exponential')

plt.tight_layout()
plt.legend()








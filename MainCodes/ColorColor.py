#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.io import fits
import pandas as pd
import os
from scipy.optimize import curve_fit

#t1,t2,t3 = np.intersect1d(T1,T2,return_indices=True)
#t1 are the common values in T1 and T3
#t2 are the indices of T1 that have a common value
#t3 are the indices of T2 that have a common value



def slope(x,x1,x2,y1,y2): #Find slope and displacement from two points
    a = (y2-y1) / (x2-x1) 
    b = (x2*y1-x1*y2) / (x2-x1)
    return a*x+b


import PIL
PIL.Image.MAX_IMAGE_PIXELS = None
def ThumbNail(Ra,Dec,objectwidth=10,w=50,Nticks = 7,check = False,pixcoord=False,addRa=[],addDec=[],addSize=[]): #w = extension to each side in pixels
    def RaDeStr(Ra,Dec):
        x = [];y = []
        x.append(np.floor(Ra*24/360))
        remainder = (Ra*24/360-np.floor(Ra*24/360))*60
        x.append(np.floor(remainder))
        remainder = (remainder - np.floor(remainder))*60
        x.append(remainder)
        str1 = "%ih %im %.4gs" % (x[0],x[1],x[2])
        y.append(np.floor(Dec))
        remainder = (Dec - np.floor(Dec))*60
        y.append(np.floor(remainder))
        remainder = (remainder - np.floor(remainder))*60
        y.append(remainder)
        str2 = "%i°%i\'%.4g\'" % (y[0],y[1],y[2])
        return str1,str2
    def coord(Ra,Dec,roundoff=pixcoord): #Transform from Ra,Dec to pixel coordinates
        [x1,y1]    = [254.,         42.]     ; [x2,y2]    = [17572.,       10651.]
        [Ra1,Dec1] = [150.7691667, 2.8131500] ; [Ra2,Dec2] = [149.3247375, 1.9292861]
        x = slope(Ra,Ra1,Ra2,x1,x2)
        y = slope(Dec,Dec1,Dec2,y1,y2)
        if roundoff:
            return x,y
        else:
            return int(np.round(x)),int(np.round(y)) #Only return integers
    def coord2(x,y): #Transform from Ra,Dec to pixel coordinates
        [x1,y1]    = [254.,         42.]     ; [x2,y2]    = [17572.,       10651.]
        [Ra1,Dec1] = [150.7691667, 2.8131500] ; [Ra2,Dec2] = [149.3247375, 1.9292861]
        Ra = slope(x,x1,x2,Ra1,Ra2)
        Dec= slope(y,y1,y2,Dec1,Dec2)
        return Ra,Dec
    
    x,y = coord(Ra,Dec)
    XMAX = 17940; YMAX = 11408 
    if check:
        if x > XMAX or x < 0 or y > YMAX or y < 0:
            return False
        else:
            return True
    elif pixcoord:
        if x > XMAX or x < 0 or y > YMAX or y < 0:
            raise Warning("Coordinates out of bounds of image")
        return y,x
    else:
        if x > XMAX or x < 0 or y > YMAX or y < 0:
            raise Warning("Coordinates out of bounds of image")
        plt.figure(figsize=(10,8),dpi=100)
        if 'image' not in globals():
            global image
            image = plt.imread('../Thumbnailer/UVISTA_YJK_large.jpg')
            print("Image not yet loaded, loading (rip ram, haha)")
        #else:
            #print("image was in locals, skipping read")
            
        
        ax = plt.subplot(111)
        xmin = x-w; xmax = x+w; ymin = y-w; ymax = y+w
        [Wx,Wy] = [w,w]
        if xmin < 0: xmin = 0       ; Wx = x
        if ymin < 0: ymin = 0       ; Wy = y
        if xmax > XMAX: xmax = XMAX
        if ymax > YMAX: ymax = YMAX
        
        ra2,dec1 = coord2(xmin,ymax)
        ra1,dec2 = coord2(xmax,ymin)
        
        rs=[];ds=[]
        RaPos = np.linspace(ra1,ra2,Nticks)
        DecPos = np.linspace(dec1,dec2,Nticks)
        for i in range(Nticks):
            ra,de = RaDeStr(RaPos[i],DecPos[i])
            rs.append(ra)
            ds.append(de)
        ax.imshow(image[ymin:ymax,xmin:xmax])
        draw_circle = plt.Circle( (Wx, Wy), radius=objectwidth,fill=False,color='r')
        ax.add_artist(draw_circle)
        if len(addRa)>0:
            if len(addRa) != len(addDec):
                raise Warning('Length of Re and Dec addition not the same')
            for i in range(len(addRa)):
                xe,ye = coord(addRa[i],addDec[i])
                if len(addSize) > 0:
                    size = addSize[i]
                else:
                    size = objectwidth
                if True:
                    draw_circle = plt.Circle( (xe-xmin, ye-ymin), radius=size,fill=False,color='r')
                    ax.add_artist(draw_circle)
        
        plt.xticks(np.linspace(xmax-xmin,0,Nticks),rs)
        plt.yticks(np.linspace(ymax-ymin,0,Nticks),ds)
        #ax.axis('off')
        plt.tight_layout()
        plt.show()
        
def ThumbNail2(RA,DEC,size=10,width=500,addRA=[],addDEC=[],addSize=[],pixcoord=False):
    if width > 4000:
        
        print('A window larger than 4000x4000 pixels is not recommended')
        cont = input('do you want to continue anyways? [Y/N]')
        if cont in ['N','n','no']:
            return
    width //= 2
    from astropy.wcs import wcs
    from astropy.nddata import Cutout2D
    filters = ['Ks','NB118','Y']
    data = {}
    for i in range(len(filters)):
        data[filters[i]] = fits.open('..\RawData\%s.fits'%filters[i])[0].data
    hdu = fits.open(r'..\RawData\NB118.fits')[0]
    w = wcs.WCS(hdu.header)
    coord = np.array([[RA],[DEC]]).T
    pix = np.round(w.wcs_world2pix(coord,True)[0]).astype(int)
    if pixcoord:
        return pix
    sel = [pix[1]-width,pix[1]+width,pix[0]-width,pix[0]+width]
    bal = [255,220,255]
    
    R = Cutout2D(data[filters[0]], position=pix, size=width*2, wcs=w).data
    G = Cutout2D(data[filters[1]], position=pix, size=width*2, wcs=w).data
    B = Cutout2D(data[filters[2]], position=pix, size=width*2, wcs=w).data
    hdu.header.update(Cutout2D(data['Y' ], position=pix, size=width, wcs=w).wcs.to_header())
    
    
    cut = [1e-1,500]
    R[np.where(R < cut[0])] = cut[0]
    G[np.where(G < cut[0])] = cut[0]
    B[np.where(B < cut[0])] = cut[0]
    
    R[np.where(R > cut[1])] = cut[1]
    G[np.where(G > cut[1])] = cut[1]
    B[np.where(B > cut[1])] = cut[1]
    
    image = np.zeros([sel[1]-sel[0],sel[3]-sel[2],3])
    image[:,:,0] = (np.log10(R) - np.log10(cut[0]) )
    image[:,:,1] = (np.log10(G) - np.log10(cut[0]) )
    image[:,:,2] = (np.log10(B) - np.log10(cut[0]) )
    
    for i in range(3):
        image[:,:,i] *= bal[i]/np.max(image[:,:,i])
    image = np.floor(image).astype(int)
    

    fig = plt.figure(figsize=(11,10),dpi=100)
    axes = fig.add_subplot(111,projection=wcs.WCS(hdu.header))
    plt.imshow(image,origin='lower')#,interpolation='gaussian')
    
    draw_circle = plt.Circle((width,width), radius=size,fill=False,color='r')
    axes.add_artist(draw_circle)
    
    if len(addRA)>0:
        longcoord = np.array([addRA,addDEC]).T
        longpix = np.round(w.wcs_world2pix(longcoord,True)).astype(int)
        if len(addSize) == 0:
            addSize = np.ones(len(addRA))*size
            
        for i in range(len(addRA)):
            draw_circle = plt.Circle( (longpix[i][0]-pix[0]+width,longpix[i][1]-pix[1]+width), radius=addSize[i],fill=False,color='r')
            axes.add_artist(draw_circle)
    
    plt.xlabel('RA')
    plt.ylabel('Dec')
    plt.show()

def nearest_index(array, value): #Find index of element closest to given value
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
def zTarget(target,sigma_target=None): #All in angstrom
    nb = 11850
    snb = 20
    nbw = 112.18
    sout = np.zeros([2,len(target)])
    if sigma_target == None:
        st = 0
    else:
        st = sigma_target
    z = nb/target - 1
    sz = np.sqrt( (st**2 * nb**2 + snb**2 * target**2) / (target**4) )
    zh = nb/(target-nbw/2) - 1
    zl = nb/(target+nbw/2) - 1
    sout[0] = z - (zl + sz)
    sout[1] = zh + sz - z
    return z,sout.T

#                Lalpha       OII     Hdelta  Hgamma  Hbeta   OIII    OIII    Halpha
#zname = [        'Lalpha'   ,'OII'  ,'Hdelta','Hgamma','Hbeta','OIII','OIII','Halpha']
#emit = np.array([1215.022732, 3727.3, 4102.8, 4340.0, 4861.3, 4959.0, 5006.8, 6562.8])
Lines2 = pd.read_csv('Lines2.csv',delimiter=';') ; [emit,zname] = [Lines2['w'],Lines2['line']]
#[emit,zname] = [np.array(pd.read_csv('lines.csv')['w']),pd.read_csv('lines.csv')['line']]
redshifts,sigma_redshifts = zTarget(emit)


### LOADING DATA ###
if 'hdu' not in locals():
    print("COSMOS data not loaded, loading...")
    #hdu = fits.open('COSMOS2020_FARMER_v1.5.fits')
    #hdr = hdu[1].header
    hdu = fits.open(r"COSMOS2020_CLASSIC_v1.8.1_formatted.fits")
    hdr = hdu[1].header
    
else:
    print("COSMOS data already loaded, skipping read")
if 'zphot' not in locals():
    print("FARMER data not loaded, loading...")
    #zphot = pd.read_csv('zphot_tractor.out',sep=" ",skiprows=52)
    hduz = fits.open(r"photoz_cosmos2020_lephare_classic_v1.out.fits.gz")
    hdrz = hduz[1].header
    zphot = hduz[1].data
else:
    print("FARMER data already loaded, skipping read")




def xysize(cut):   #Determine the radius in pixels around the central pixel that detection was performed
    a = 40000**2/(17940*11408)
    npix = hdu[1].data['FLUX_RADIUS'][cut]
    return np.sqrt(npix/(np.pi*a))



def RaDec(entries): #returns the J2000 coordinates for a given indices of COSMOS data
    Ra  = np.array(hdu[1].data['ALPHA_J2000'])
    Dec = np.array(hdu[1].data['DELTA_J2000'])

    outp = np.zeros([len(entries),2])
    c = 0
    for i in entries:
        outp[c] = np.array([Ra[i],Dec[i]])
        c += 1
    return outp

def corr(cut,err=False):
    nb = hdu[1].data['UVISTA_NB118_MAG_AUTO'][cut]
    j  = hdu[1].data['UVISTA_J_MAG_AUTO'][cut]
    y  = hdu[1].data['UVISTA_Y_MAG_AUTO'][cut]
    nbr = hdu[1].data['UVISTA_NB118_MAGERR_AUTO'][cut]
    jr  = hdu[1].data['UVISTA_J_MAGERR_AUTO'][cut]
    yr  = hdu[1].data['UVISTA_Y_MAGERR_AUTO'][cut]
    out = j-nb + 0.07
    outerr = np.sqrt( (1089*jr**2 + 2500*nbr**2 + 289*yr**2) / (50))
    
    cond1 = np.where(y-j <= 0.45)
    cond2 = np.where(y-j >  0.45)
    
    out[cond1] = (j[cond1] - nb[cond1]) + 0.34*(y[cond1]-j[cond1])
    out[cond2] = (j[cond2] - nb[cond2]) + 0.153
    outerr[cond1] = np.sqrt(jr[cond1]**2 + nbr[cond1]**2)
    outerr[cond2] = np.sqrt(jr[cond2]**2 + nbr[cond2]**2)
    if err:
        return out, outerr
    else:
        return out
#%%
def data(sel,typesel='MAG_AUTO'):

    global NB118,Y,Z,J,H,Ks,B,V,z
    NB118 = hdu[1].data['UVISTA_NB118_'+typesel][sel]+0.85
    Y     = hdu[1].data['UVISTA_Y_'    +typesel][sel]+0.61
    J     = hdu[1].data['UVISTA_J_'    +typesel][sel]+0.92
    H     = hdu[1].data['UVISTA_H_'    +typesel][sel]+1.38
    Ks    = hdu[1].data['UVISTA_Ks_'   +typesel][sel]+1.84
    B     = hdu[1].data['SC_B_' +typesel][sel]
    V     = hdu[1].data['SC_V_' +typesel][sel]
    z     = hdu[1].data['HSC_z_' +typesel][sel]

plt.close('all')
aper = '_AUTO'
ranged = np.where( (hdu[1].data['UVISTA_Y_MAGERR'+aper ]    <= 10) & (hdu[1].data['UVISTA_Y_MAG'+aper ]     >= 0)
                 & (hdu[1].data['UVISTA_J_MAGERR'+aper ]    <= 10) & (hdu[1].data['UVISTA_J_MAG'+aper ]     >= 0)
                 & (hdu[1].data['UVISTA_H_MAGERR'+aper ]    <= 10) & (hdu[1].data['UVISTA_H_MAG'+aper ]     >= 0)
                 & (hdu[1].data['UVISTA_Ks_MAGERR'+aper]    <= 10) & (hdu[1].data['UVISTA_Ks_MAG'+aper]     >= 0)
                 )[0]

data(ranged,'MAG'+aper)
ID = hdu[1].data['ID'][ranged]

Area = 1.5*1.2

#--------Just cecking for a random redshift if the ID and position are the same-----
# id1 = zphot['Id'][np.where(zphot['zBEST']>7)[0][0]]
# zpos = [zphot['alpha'][np.where(zphot['zBEST']>7)[0][0]],zphot['delta'][np.where(zphot['zBEST']>7)[0][0]]]
# i2 = np.where(hdu[1].data['ID'] == id1)[0][0]
# photpos = [hdu[1].data['ALPHA_J2000'][i2],hdu[1].data['DELTA_J2000'][i2]]


#----Narrowband excess plot----

plt.figure(figsize=(12,8),dpi=100)
plt.plot(Y-J,J-NB118,'ok',ms=1)
plt.xlim([-1,2])
plt.ylim([-1,2])
plt.xlabel(r'$(Y-J)$ [mag]')
plt.ylabel(r'$(J-NB118)$ [mag]')
plt.tight_layout()


#-----Narrowband excess histogram (2D)------

histsele = np.where( (Y-J >= -1) & (Y-J <= 2)
                    & (J-NB118 >= -2) & (J-NB118 <= 2) 
                    )

plt.figure(figsize=(12,8),dpi=100)
h,xe,ye = np.histogram2d( (Y-J)[histsele],(J-NB118)[histsele],500)
plt.imshow(np.rot90(h)+1e-10,extent=(-1,2,-2,2),aspect='auto',norm=LogNorm(vmin=0.1),cmap='binary')
plt.colorbar()
#plt.xlim([-1,2])
#plt.ylim([-1,2])
plt.xlabel(r'$(Y-J)$ [mag]')
plt.ylabel(r'$(J-NB118)$ [mag]')
plt.tight_layout()





#------2D Hist of Y-J vs H-Ks------
extent = (-0.5,0.5,-1,0.4)




xcut = np.linspace(extent[0],.2,10)
cutpoints = (-.3,0,-.7,-.5)
magStarCut = 24.5
cut = slope(xcut,*cutpoints)

histsele = np.where(  (Y - J  >= extent[0]) & (Y - J  <= extent[1])
                    & (H - Ks >= extent[2]) & (H - Ks <= extent[3])
                    & (Ks > 0) & (Ks < magStarCut)

                    )[0]
print('Total of %i sources' % len(histsele))
fig1 = plt.figure(figsize=(5.5,4.5),dpi=100)
h,xe,ye  = np.histogram2d( (Y - J)[histsele],(H - Ks)[histsele],300,[[extent[0],extent[1]],[extent[2],extent[3]]])
plt.imshow(np.rot90(h)+1e-10,extent=extent,aspect='auto',norm=LogNorm(vmin=1,),cmap='binary')
plt.colorbar()
plt.plot(xcut,cut,'--r',label='Star cut')
plt.xlabel(r'$(Y-J)$ [mag]')
plt.ylabel(r'$(H-Ks)$ [mag]')
#plt.title('Star selection')
#plt.legend()




starsele = histsele[np.where( ( (H-Ks)[histsele] <= slope( (Y-J)[histsele],*cutpoints ) )
                             & (((Y-J)[histsele] >= -0.5) & ((Y-J)[histsele] <= .2))
                             )[0]]
print('Based on YJHKs selection, %i stars were found' % len(starsele))
h,xe,ye  = np.histogram2d( (Y - J)[starsele],(H - Ks)[starsele],300,[[extent[0],extent[1]],[extent[2],extent[3]]])
alpha = np.zeros(h.shape)
alpha[np.where(h>=1)] = .3
plt.imshow(np.rot90(h)+1e-10,extent=extent,aspect='auto',norm=LogNorm(vmin=1,),cmap='Reds',alpha=np.rot90(alpha))
plt.xlim([extent[0],extent[1]])
plt.ylim([extent[2],extent[3]])


fig1.subplots_adjust(top=0.987,bottom=0.106,left=0.121,right=1.0,hspace=0.2,wspace=0.2)
#plt.savefig('../Thesis/fig/YJHKscut.eps')

#----- Number of sources histogram----

HistNames = os.listdir('.\HistData')
marks = ['or','sg','hm','^b']
leg = ['Bielby et al. 2010','McCracken et al. 2010','Quadri et al. 2007','McCracken H. J., et al., 2012']


starsele = np.where( ( (H-Ks) <= slope( (Y-J),*cutpoints ) )
                             & (((Y-J) >= -0.5) & ((Y-J) <= 0.2))
                             & ((H-Ks) >= -1) 
                             & (Ks > 0) & (Ks < magStarCut)
                             )[0]


nostar1 = np.where(  (Ks>15) & (Ks<28) 
                   )[0]

nostar = np.setdiff1d(nostar1,starsele)

fig1 = plt.figure(figsize=(6,4),dpi=200)
fig1.subplots_adjust(top=0.998,bottom=0.112,left=0.103,right=0.997,hspace=0.2,wspace=0.2)
ax = plt.subplot(111)
ax.grid(which='both',color='grey',alpha=.2)


#h,be = np.histogram(Ks[nostar],np.linspace(19.50,27,16))
h,be = np.histogram(Ks[nostar],np.linspace(19.50,29,20))
bc = be[:-1] + 0.5*(be[1:]-be[:-1])
plt.plot(bc-1.84,2*h/Area * 0.5/(be[1]-be[0]),'8k',label='This work')
print("%i galaxies in histogram" % np.sum(h) )
for i in range(len(HistNames)):
    daa = np.loadtxt(r'.\HistData\%s' % (HistNames[i]),skiprows=1,delimiter=', ')
    plt.plot(daa[:,0],daa[:,1]*2,marks[i],fillstyle='none',label=leg[i])
plt.ylim([50,4e+5])
plt.xlim([17.5,26.1])
plt.legend()
plt.tick_params(axis='y',which='both')
plt.yscale('log')
plt.xlabel(r'$K_s$ [mag]')
plt.ylabel(r'$N_{gal} \ [$mag$^{-1}$deg$^{-2}$]')
#plt.savefig('../Thesis/fig/NgalHistogram.pdf')



Filters = pd.read_csv('Filters.csv',sep = ', ',engine='python')
def spec(cut,typesel='FLUX'+aper):
    mean = Filters['Mean']
    wavsel = np.array([32,2,33,3,5,7,8,9,10,30,31])
    wav = np.array(mean[wavsel])
    
    sel = ranged[cut]
    
    #NB118 = hdu[1].data['UVISTA_NB118_'+typesel][sel]
    Y     = hdu[1].data['UVISTA_Y_'    +typesel][sel]
    J     = hdu[1].data['UVISTA_J_'    +typesel][sel]
    H     = hdu[1].data['UVISTA_H_'    +typesel][sel]
    Ks    = hdu[1].data['UVISTA_Ks_'   +typesel][sel]
    B     = hdu[1].data['SC_B_' +typesel][sel]
    V     = hdu[1].data['SC_V_' +typesel][sel]
    z     = hdu[1].data['HSC_z_' +typesel][sel]
    g     = hdu[1].data['HSC_g_' +typesel][sel]
    r     = hdu[1].data['HSC_r_' +typesel][sel]
    II1     = hdu[1].data['IRAC_CH1_' +typesel[:-5]][sel]
    II2     = hdu[1].data['IRAC_CH2_' +typesel[:-5]][sel]
    
    sp = np.array([B,g,V,r,z,Y,J,H,Ks,II1,II2])
    typesel = typesel[:-5] + 'ERR' + typesel[-5:]
    #NB118 = hdu[1].data['UVISTA_NB118_'+typesel][sel]
    Y     = hdu[1].data['UVISTA_Y_'    +typesel][sel]
    J     = hdu[1].data['UVISTA_J_'    +typesel][sel]
    H     = hdu[1].data['UVISTA_H_'    +typesel][sel]
    Ks    = hdu[1].data['UVISTA_Ks_'   +typesel][sel]
    B     = hdu[1].data['SC_B_' +typesel][sel]
    V     = hdu[1].data['SC_V_' +typesel][sel]
    z     = hdu[1].data['HSC_z_' +typesel][sel]
    g     = hdu[1].data['HSC_g_' +typesel][sel]
    r     = hdu[1].data['HSC_r_' +typesel][sel]
    II1     = hdu[1].data['IRAC_CH1_' +typesel[:-5]][sel]
    II2     = hdu[1].data['IRAC_CH2_' +typesel[:-5]][sel]
    spr = np.array([B,g,V,r,z,Y,J,H,Ks,II1,II2])
    
    return wav,sp,spr

def planck(lam,T,N):
    wave = lam/10000000000
    a = 1.191042953e-16
    b = 0.01438777354
    return ( a*N /(wave**5) ) / ( np.exp(b/(wave*T)) - 1 )

def Wien(lam):
    wave = lam/10000000000
    b = 0.002897772914
    return b/wave
    
def Lum(Temp):
    Tu = np.array([30000,30000,10000,7500,6000,5200,3700])
    Tl = np.array([30000,20000, 7500,6000,5200,3700,2400])
    T = (Tu+Tl)/2
    Lu = np.array([30000,30000,25,5,1.5,0.6,0.008])
    Ll = np.array([30000,25000,5,1.5,0.6,0.008,0.001])
    L = (Lu+Ll)/2
    return 10**np.interp(Temp,T[::-1],np.log10(L[::-1]))


plt.figure(figsize=(10,10),dpi=100)
for i in range(100):
    plt.subplot(10,10,1+i)
    wav,sp,spr = spec(starsele[i])
    plt.errorbar(wav,sp,spr,fmt='.k',label='Data')
    Guess = (Wien(wav[np.argmax(sp)]),2e-8)
    ab,cov = curve_fit(planck,wav,sp,p0=Guess,maxfev=2000)
    wav = np.linspace(min(wav),max(wav),500)
    #plt.plot(wav,planck(wav,*Guess),label='Guess')
    plt.plot(wav,planck(wav,*ab),label='Fit')
    plt.axis('off')
    # plt.legend()
    # plt.xlabel('Wavelength [Å]')
    # plt.ylabel('Flux')
plt.tight_layout()



T = []
chi = []
for i in range(len(starsele)):
    wav,sp,spr = spec(starsele[i])
    if min(sp) > 0:
        Guess = (Wien(wav[np.argmax(sp)]),2e-8)
        ab,cov = curve_fit(planck,wav,sp,sigma=spr,p0=Guess,maxfev=1000)
        T.append(ab[0])
        chi.append(np.sum( (sp - planck(wav,*ab))**2 / (spr**2) )/(len(sp)-2))
T = np.array(T)
L = Lum(T)


plt.figure(figsize=(7,5),dpi=100)
plt.hist(T,200,fc='k')
plt.xlabel('Temperature [T]')





extent = (3,-1,-3,0)
plt.figure(figsize=(12,8),dpi=100)
plt.plot((B-V)[starsele],(z-Ks)[starsele],'.k',ms=1)
plt.xlim([extent[0],extent[1]])
plt.ylim([extent[2],extent[3]])


ks1 = starsele[np.where( (Ks[starsele]-1.84 > 18.75) & (Ks[starsele]-1.84 <19.0) )[0]]
ks2 = starsele[np.where( (Ks[starsele]-1.84 > 19.50) & (Ks[starsele]-1.84 <20.5) )[0]]

extent = (0,6,-3,0)
plt.figure(figsize=(7,4),dpi=100)
#plt.plot((B-z)[starsele],(z-Ks)[starsele],'.k',ms=1,label=r'$YJHK_s$ Selected stars')
plt.plot((B-z)[ks1],(z-Ks)[ks1],'sb',ms=5,fillstyle='none',label=r'$18.75<K_s<19$',zorder=10,alpha=.4)
plt.plot((B-z)[ks2],(z-Ks)[ks2],'.r',ms=1,fillstyle='none',label=r'$19.5<K_s<20.5$')
plt.xlim([extent[0],extent[1]])
plt.ylim([extent[2],extent[3]])
plt.legend()
plt.xlabel(r'$(B-z)$ [mag]')
plt.ylabel(r'$(z-K_s)$ [mag]')
plt.tight_layout()





IDz = np.array(zphot['Id'][np.where( zphot['zBEST'] >= 0.4)[0]])
zsel = np.intersect1d(ID,IDz,return_indices=True)[1]
zsel = np.intersect1d(zsel,np.where(Ks<26)[0])
nostar = np.intersect1d(nostar,zsel)



extent = (0,5,-3,2)
plt.figure(figsize=(20,8),dpi=100)
ax = plt.subplot(121)
plt.hist2d((B-z),(z-Ks),200,[[extent[0],extent[1]],[extent[2],extent[3]]],norm=LogNorm(vmin=1,),cmap='binary')
plt.colorbar()

plt.xlim([extent[0],extent[1]])
plt.ylim([extent[2],extent[3]])
plt.xlabel(r'$(B-z)$ [mag]')
plt.ylabel(r'$(z-Ks)$ [mag]')
plt.title('All sources')



ax = plt.subplot(122)
plt.hist2d((B-z)[nostar],(z-Ks)[nostar],200,[[extent[0],extent[1]],[extent[2],extent[3]]],norm=LogNorm(vmin=1,),cmap='binary')
plt.colorbar()

plt.xlim([extent[0],extent[1]])
plt.ylim([extent[2],extent[3]])
plt.xlabel(r'$(B-z)$ [mag]')
plt.ylabel(r'$(z-Ks)$ [mag]')
plt.title('Stars subtracted')
plt.tight_layout()





g = hdu[1].data['HSC_g_MAG'+aper][ranged]
r = hdu[1].data['HSC_r_MAG'+aper][ranged]
I1 = hdu[1].data['IRAC_CH1_MAG'][ranged]
I2 = hdu[1].data['IRAC_CH2_MAG'][ranged]

kc = np.where( (Ks<24.5) & (I1-I2 != 0) )[0]
kcs = np.intersect1d(kc,starsele)



extent = (-2,1,-.5,2.5)
plt.figure(figsize=(10,6),dpi=100)
h1 = np.histogram2d((J-Ks)[kc],(g-r)[kc],200,[[extent[0],extent[1]],[extent[2],extent[3]]])[0]
h2 = np.histogram2d((J-Ks)[kcs],(g-r)[kcs],200,[[extent[0],extent[1]],[extent[2],extent[3]]])[0]
alpha = np.zeros(h2.shape)
alpha[np.where(h2>=2)] = .3

plt.imshow(np.rot90(h1),norm=LogNorm(vmin=1,),extent=extent,cmap='binary',aspect='auto')
plt.colorbar()
plt.imshow(np.rot90(h2),norm=LogNorm(vmin=1,),extent=extent,cmap='Reds',alpha=np.rot90(alpha),aspect='auto')
plt.xlabel(r'$(J-K_s)$ [mag]')
plt.ylabel(r'$(g-r)$ [mag]')
plt.tight_layout()


extent = (-1.75,.5,-1,.75)
plt.figure(figsize=(10,6),dpi=100)
h1 = np.histogram2d((J-Ks)[kc],(I1-I2)[kc],200,[[extent[0],extent[1]],[extent[2],extent[3]]])[0]
h2 = np.histogram2d((J-Ks)[kcs],(I1-I2)[kcs],200,[[extent[0],extent[1]],[extent[2],extent[3]]])[0]
alpha = np.zeros(h2.shape)
alpha[np.where(h2>=1)] = .3

plt.imshow(np.rot90(h1),norm=LogNorm(vmin=1,),extent=extent,cmap='binary',aspect='auto')
plt.colorbar()
plt.imshow(np.rot90(h2),norm=LogNorm(vmin=1,),extent=extent,cmap='Reds',alpha=np.rot90(alpha),aspect='auto')
plt.xlabel(r'$(J-K_s)$ [mag]')
plt.ylabel('(I1-I2) [mag]')
plt.tight_layout()





#%%

#-----------------------------------------------------------------------
#----------Narrowband excess vs narrowband magnitude--------------------
#-----------------------------------------------------------------------


JNBcorr,JNBcorrerr = corr(ranged,err='both')

[xc,yc] = [[18,28],[-.5,5]]

plt.close(fig=19)
plt.figure(19,figsize=(20,12),dpi=80)
plt.plot(NB118,JNBcorr,'ok',ms=1)
plt.xlim(xc)
plt.ylim(yc)
plt.xlabel(r'$NB118$ [mag]')
plt.ylabel(r'$(J-NB118)_{corr}$ [mag]')
plt.tight_layout()




def expcut(x,mu=28,sig=1.5,N1=2.5,N2=0.25): #mu=25.5,sig=0.6,N1=2.5,N2=0.25
    a = N1/(sig*np.sqrt(2*np.pi))
    return a * np.exp(- (x-mu)**2/(2*sig**2))+N2
    # return np.ones(len(x))*0.2

sele = np.where( (NB118 >= xc[0]) & (NB118 <= xc[1]) 
               & (JNBcorr >= expcut(NB118))
                )[0]
plt.plot(np.linspace(xc[0],xc[1],500),expcut(np.linspace(xc[0],xc[1],500)),'--r',label='Selection')
plt.plot(NB118[sele],JNBcorr[sele],'ob',ms=1)

plt.plot(NB118[starsele],JNBcorr[starsele],'or',ms=1,label='Stars')



z1 = np.array([0.238,0.301,0.654,0.753,0.793,1.357,1.428,2.165,3.211,4.00,8.72])
z2 = np.array([0.260,0.324,0.683,0.788,0.834,1.422,1.471,2.224,3.297,4.02,8.78])
red = (z2+z1)/2
z12name = ['S III', 'S III', 'Ar III', 'S II', r'H$\alpha$ + N II', 'O III', r'H$\beta$', 'O II', 'Mg II', 'Fe II', r'L$\alpha$']





zpoints = []
for zi in range(len(red)):
    temp = []
    zphotwhere = np.where( (zphot['zBEST'] >= z1[zi]) & (zphot['zBEST'] <= z2[zi]) 
                          & ( np.abs(zphot['zPDF_u68']-zphot['zPDF_l68']) <= 10 )
                          )[0]
    IDENT = np.array(zphot['Id'][zphotwhere])
    #We want to find the index of COSMOS data that has pecific redshifts
    temp = sele[np.intersect1d(IDENT,ID[sele],return_indices=True)[-1]]
    print('Found %5i sources at redshift %.4g' % (len(temp),red[zi]))
    zpoints.append(temp)



for i in range(len(red)): #len(red)
    if len(zpoints[i]) > 0:
        if red[i] < 1:
            plt.plot(NB118[zpoints[i]],JNBcorr[zpoints[i]],'o',ms=10,fillstyle='none',label='%.4i at z = %.2g  %s' % (len(zpoints[i]),red[i],z12name[i]))
        else:
            plt.plot(NB118[zpoints[i]],JNBcorr[zpoints[i]],'o',ms=10,fillstyle='none',label='%.4i at z = %.3g  %s' % (len(zpoints[i]),red[i],z12name[i]))
plt.legend()
plt.tight_layout()
plt.savefig('ColorMag.png',dpi=500)

#%%


Ra = []
Dec = []
addSize = []
for i in range(len(zpoints)):
    Ra.append( hdu[1].data['ALPHA_J2000' ][ranged[zpoints[i]]])
    Dec.append(hdu[1].data['DELTA_J2000'][ranged[zpoints[i]]])
    addSize.append(xysize(ranged[zpoints[i]]))
plt.close(fig = 20)
plt.figure(20,figsize=(16,12),dpi=100)
for i in range(len(zpoints)):
    plt.subplot(3,4,1+i)
    plt.hist2d(Ra[i],Dec[i],100)
    plt.title(z12name[i])
    plt.gca().invert_xaxis()
plt.tight_layout()

zs = 5
#ThumbNail(150.1,2.0,0,1000,Nticks=7,addRa=Ra[zs],addDec=Dec[zs],addSize=addSize[zs]);plt.title(z12name[zs])
ThumbNail2(149.4,2.11,0,4000,addRA=Ra[zs],addDEC=Dec[zs],addSize=addSize[zs]*20);plt.title(z12name[zs])





#%%

zrange = [0.1,9]
zsigma = 0.1

photcut = np.where( (zphot['zBEST'] >= zrange[0]) & (zphot['zBEST'] <= zrange[1]) 
                   & ( np.abs(zphot['zPDF_u68']-zphot['zPDF_l68']) <= zsigma )
                   & ( zphot['zPDF_u68'] > -10 ) 
                   & ( zphot['zPDF_l68'] > -10 )
                   )[0]
IDz = zphot['Id'][photcut]
IDc = np.where( JNBcorr < 100 )[0]
ID2 = ID[IDc]

com,t1,t2 = np.intersect1d(ID2,IDz,return_indices=True)
tz = photcut[t2]
t1 = IDc[t1]

z = zphot['zBEST'][tz]
zs = np.array( [ zphot['zBEST'][tz]-zphot['zPDF_l68'][tz] , zphot['zPDF_u68'][tz]-zphot['zBEST'][tz] ])

plt.close(fig = 30)
plt.figure(30,figsize=(12,8),dpi=100)
ax = plt.subplot(111)

plt.errorbar(z,JNBcorr[t1],xerr=zs,fmt='ok',ms=2,capsize=0)
plt.ylim([0,np.max(JNBcorr[t1])])
plt.xlabel(r'redshift')
plt.ylabel(r'$(J-NB118)_{corr}$ [mag]')
plt.tight_layout()

#%%

zsig = np.abs(zphot['zPDF_u68'][tz]-zphot['zPDF_l68'][tz])

plt.close(fig = 40)
plt.figure(40,figsize=(12,8),dpi=100)
plt.hist(z,200,facecolor='k')

#%%
phot = []
for i in range(len(hdr)):
    search = 'MAG_APER3'
    if search in "%s" % hdr[i] and 'ERR' not in "%s" % hdr[i]and '_IA' not in "%s" % hdr[i]and '_IB' not in "%s" % hdr[i]:
        print("%s" % hdr[i])
        phot.append(hdr[i])
#%%
depth = []
for i in range(len(phot)):
    sel = np.where(hdu[1].data[phot[i][:-9]+'FLUX_APER3'] >= 3*hdu[1].data[phot[i][:-9]+'FLUXERR_APER3'])[0]
    depth.append(np.max(hdu[1].data[phot[i]][sel]))



plt.close(fig = 41)
plt.figure(41,figsize=(12,8),dpi=100)
for i in range(len(depth)):
    plt.plot(i,depth[i],'o',label=phot[i][:-10])
plt.gca().invert_yaxis()
plt.legend(ncol=2)
plt.tight_layout()






















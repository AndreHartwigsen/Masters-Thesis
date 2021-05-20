import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
from astropy.io import fits
import pandas as pd
from astropy.visualization import make_lupton_rgb
from astropy.wcs import wcs
from astropy.nddata import Cutout2D
import sep
from matplotlib.patches import Ellipse
import warnings
warnings.filterwarnings("ignore")





RA_pix_scale  = 4.166062362287448e-05 #Degrees per pixel
DEC_pix_scale = 4.165047454396387e-05 #Degrees per pixel

y_pix = np.array([ 8700, 14070, 19440, 24810, 30180, 35550])
x_pix = np.array([13431, 22248, 31115, 39900])

pixxers = np.zeros([len(x_pix)*len(y_pix),2])
p = 0
for i in range(len(x_pix)):
    for i2 in range(len(y_pix)):
        pixxers[p] = np.array([x_pix[i],y_pix[i2]])
        p += 1

def sort(arrs,direction = 'asc'):
    arrs = [np.asarray(x) for x in arrs]
    inds = np.argsort(arrs[0])
    if direction != 'asc':
        inds = inds[::-1]
    return [x[inds] for x in arrs]
def smooth(x,Nm):
    Nm += (Nm % 2)
    y = np.zeros(len(x))
    na = np.int(np.floor(Nm/2))
    for i in range(len(x)):
        if i<na:
            y[i] = np.mean(x[:i+na+1])
        if i>na-1 and i<len(x)-na:
            y[i] = np.mean(x[i-na:i+na+1])
        if i>len(x)-na-1:
            y[i] = np.mean(x[i-na:])
    return y

def subidex(n): #Find size of subplots closest to square to have n subplots
    a = np.ceil(np.sqrt(n)).astype(int)
    b = np.ceil(n/a).astype(int)
    return b,a

plt.close('all')

Optimal_coord = np.array([[39900],[11860]])
Optimal_width = 5937//2



RA = 150.13569952483107
DEC = 2.2
width = 3000


#%%

for ipos in range(len(pixxers)):
    if 'Objects%i.npy'%ipos not in os.listdir('.//SavedObjectsNpy/'):
        hdu = fits.open(r'..\RawData\NB118.fits')[0]
        w = wcs.WCS(hdu.header)
        RA,DEC = w.wcs_pix2world(pixxers,True)[ipos]
        coord = np.array([[RA],[DEC]]).T
        print(coord,RA,DEC,'%.3g%% done'%(100*ipos/len(pixxers)))
        pix = np.round(w.wcs_world2pix(coord,True)[0]).astype(int)
        hdu.header.update(Cutout2D(fits.open(r'..\RawData\NB118.fits')[0].data, position=pix, size=width*2, wcs=w).wcs.to_header())
        
        
        def Import(Filter):
            data = np.array(Cutout2D(fits.open(r'..\RawData\%s.fits'%Filter)[0].data, position=pix, size=width*2, wcs=w).data,order='c')
            weight = np.array(Cutout2D(fits.open(r'..\RawData\%s_weights.fits'%Filter)[0].data, position=pix, size=width*2, wcs=w).data,order='c')
            data = data.byteswap().newbyteorder()
            error = 1/( (weight.byteswap().newbyteorder())**2 )
            return data,error
        
        nbdata,error = Import('NB118')
        sep.set_extract_pixstack(1e+7)
        sep.set_sub_object_limit(4096)
        
        #bkg = sep.Background(nbdata,bw=128,bh=128)
        objects = sep.extract(nbdata, 2.5,minarea=5,err = error,deblend_cont= 0.00001)
        np.save('.//SavedObjectsNpy/Objects%i'%ipos,objects)


#%%


# RA = 149.40000898
# DEC = 1.78496597
# width = 5937//2



hdu = fits.open(r'..\RawData\NB118.fits')[0]
w = wcs.WCS(hdu.header)
coord = np.array([[RA],[DEC]]).T
pix = np.round(w.wcs_world2pix(coord,True)[0]).astype(int)
hdu.header.update(Cutout2D(fits.open(r'..\RawData\NB118.fits')[0].data, position=pix, size=width*2, wcs=w).wcs.to_header())


def Import(Filter):
    data = np.array(Cutout2D(fits.open(r'..\RawData\%s.fits'%Filter)[0].data, position=pix, size=width*2, wcs=w).data,order='c')
    weight = np.array(Cutout2D(fits.open(r'..\RawData\%s_weights.fits'%Filter)[0].data, position=pix, size=width*2, wcs=w).data,order='c')
    data = data.byteswap().newbyteorder()
    error = 1/( (weight.byteswap().newbyteorder())**2 )
    return data,error

nbdata,error = Import('NB118')

plt.close(fig=1)
plt.close(fig=2)

#%%
fig1 = plt.figure(figsize=(10,10),dpi=100)
ax = plt.subplot(111,projection=wcs.WCS(hdu.header))
plt.imshow(nbdata,norm=LogNorm(vmin=1,vmax=30),cmap='binary')
plt.xlabel('Ra')
plt.ylabel('Dec')
fig1.subplots_adjust(top=0.985,bottom=0.059,left=0.083,right=0.985,hspace=0.2,wspace=0.2)


a1 = smooth(np.sum(fits.open(r'..\RawData\%s.fits'%'NB118')[0].data,axis=0),501)
a2 = smooth(np.sum(fits.open(r'..\RawData\%s.fits'%'NB118')[0].data,axis=1),501)

fig2 = plt.figure(figsize=(20,10),dpi=100)
Ra1,De1 = wcs.WCS(fits.open(r'..\RawData\NB118.fits')[0].header).pixel_to_world_values((0,len(a1)),(0,len(a2)))
plt.subplot(1,2,1)
plt.plot(np.linspace(Ra1[0],Ra1[1],len(a1)),a1)
plt.subplot(1,2,2)
plt.plot(np.linspace(De1[0],De1[1],len(a2)),a2)
plt.tight_layout()







cutsRa = []
for i in range(len(a1)):
    if a1[i] >= 2000 and a1[i-1] < 2000:
        cutsRa.append(i)
    if a1[i] <= 2000 and a1[i-1] > 2000:
        cutsRa.append(i)
cutsDec= np.where(a1 >= 2000)[0]
cutsDec = [cutsDec[0],cutsDec[-1]]*4

NB188CUTS = wcs.WCS(fits.open(r'..\RawData\NB118.fits')[0].header).pixel_to_world_values(cutsRa,cutsDec)

for i in range(4):
    print(cutsRa[2*i+1]-cutsRa[2*i])


DATA = fits.open(r'..\RawData\%s.fits'%'NB118')[0].data

plt.close(fig=2)
plt.figure(2,figsize=(10,10),dpi=100)

yhist = np.zeros([4,len(a2)])
for i in range(4):
    yhist[i] = np.sum(DATA[:,cutsRa[2*i]:cutsRa[2*i+1]],axis=1)
    ax=plt.subplot(221+i)
    hist_range = (np.where(yhist[i] > 10)[0][0],np.where(yhist[i] > 10)[0][-1])
    plt.hist(np.arange(len(yhist[0])),100,weights=yhist[i],range=hist_range,fc='k')
    plt.yscale('log')
plt.tight_layout()




#%%

def matmin(mat,above=1):
    y = []
    val = []
    for i in range(len(mat)):
        x = np.arange(len(mat[i]))
        sel = np.where(mat[i] > above)[0]
        i_min = np.argmin(mat[i][sel])
        y.append(x[sel][i_min])
        val.append(np.min(mat[i][sel]))
    return y,np.argmin(val)
plt.close(fig=2)
plt.figure(2,figsize=(10,10),dpi=100)
score = np.zeros([4,len(a2)])
RaLoc = np.zeros(4)

for i in range(4):
    plt.subplot(2,2,1+i)
    hist_range = (np.where(yhist[i] > 10)[0][0],np.where(yhist[i] > 10)[0][-1])
    
    Wide = cutsRa[2*i+1]-cutsRa[2*i]
    RaLoc[i] = cutsRa[2*i]+Wide//2
    print(hist_range,Wide)
    for ih in range(hist_range[0]+Wide//2,hist_range[1]-Wide//2):
        score[i][ih] = np.sum(yhist[i][ih-Wide//2:ih+Wide//2])
    plt.plot(score[i],'k')
    plt.plot(np.arange(len(score[i]))[np.where(score[i]>10)[0]],score[i][np.where(score[i]>10)[0]],'--r')
plt.tight_layout()










sep.set_extract_pixstack(1e+6)
#bkg = sep.Background(nbdata,bw=128,bh=128)
objects = sep.extract(nbdata, 2.5,minarea=5,err = error,deblend_cont= 0.00001)
np.save('.//SavedObjectsNpy/Objects%i'%0,objects)





#%%   Loading extracted data and flux extracting 
hdu = fits.open(r'..\RawData\NB118.fits')[0]
w = wcs.WCS(hdu.header)
RA,DEC = w.wcs_pix2world(pixxers,True)[0]
coord = np.array([[RA],[DEC]]).T
pix = np.round(w.wcs_world2pix(coord,True)[0]).astype(int)
hdu.header.update(Cutout2D(fits.open(r'..\RawData\NB118.fits')[0].data, position=pix, size=width*2, wcs=w).wcs.to_header())
def Import(Filter):
    data = np.array(Cutout2D(fits.open(r'..\RawData\%s.fits'%Filter)[0].data, position=pix, size=width*2, wcs=w).data,order='c')
    weight = np.array(Cutout2D(fits.open(r'..\RawData\%s_weights.fits'%Filter)[0].data, position=pix, size=width*2, wcs=w).data,order='c')
    data = data.byteswap().newbyteorder()
    error = 1/( (weight.byteswap().newbyteorder())**2 )
    return data,error

objects = np.load('.//SavedObjectsNpy/Objects%i.npy'%0)
x = objects['x']
y = objects['y']
Ra,Dec = wcs.WCS(hdu.header).pixel_to_world_values(x,y)

a,b,theta = objects['a'], objects['b'], objects['theta']



filt = ['Y','NB118','J','H','Ks']
FLUX = {}
FLUXERR = {}
FLAG = {}



# nbdata,error = Import('NB118')
# kron_rad,kron_flag = sep.kron_radius(nbdata, x, y, a, b, theta,6.0)
# kron_rad[kron_rad<3.5/2.5] = 3.5/2.5
# KRON = np.copy(kron_rad)
# for i in range(len(filt)):
#     d,r = Import(filt[i])
#     bkg = sep.Background(d,bw=128,bh=128)
#     r = np.sqrt(r**2+bkg.rms()**2)
#     #FLUX[filt[i]],FLUXERR[filt[i]],FLAG[filt[i]] = sep.sum_ellipse(d-bkg, x, y, a,b,theta,2.5*kron_rad,err=r,subpix=1)
#     FLUX[filt[i]],FLUXERR[filt[i]],FLAG[filt[i]] = sep.sum_circle(d-bkg, x, y, 2.5*kron_rad,err=r,subpix=1)



for i in range(len(pixxers)-1):
    OBS = np.load('.//SavedObjectsNpy/Objects%i.npy'%(i+1))
    objects = np.concatenate((objects,OBS))
    hdu = fits.open(r'..\RawData\NB118.fits')[0]
    w = wcs.WCS(hdu.header)
    RA,DEC = w.wcs_pix2world(pixxers,True)[i+1]
    coord = np.array([[RA],[DEC]]).T
    pix = np.round(w.wcs_world2pix(coord,True)[0]).astype(int)
    hdu.header.update(Cutout2D(fits.open(r'..\RawData\NB118.fits')[0].data, position=pix, size=width*2, wcs=w).wcs.to_header())
    x = OBS['x']
    y = OBS['y']
    
    Rs1,Ds1 = wcs.WCS(hdu.header).pixel_to_world_values(x,y)
    Ra = np.concatenate((Ra,Rs1))
    Dec = np.concatenate((Dec,Ds1))
  
    #---------Flux extractor part-------
#     def Import(Filter):
#         data = np.array(Cutout2D(fits.open(r'..\RawData\%s.fits'%Filter)[0].data, position=pix, size=width*2, wcs=w).data,order='c')
#         weight = np.array(Cutout2D(fits.open(r'..\RawData\%s_weights.fits'%Filter)[0].data, position=pix, size=width*2, wcs=w).data,order='c')
#         data = data.byteswap().newbyteorder()
#         error = 1/( (weight.byteswap().newbyteorder())**2 )
#         return data,error

#     nbdata,error = Import('NB118')

#     a,b,theta = OBS['a'], OBS['b'], OBS['theta']
#     theta[np.abs(theta)>np.pi/2] = np.sign(theta[np.abs(theta)>np.pi/2])*np.pi/2
#     bkg = sep.Background(d,bw=128,bh=128)
#     kron_rad,kron_flag = sep.kron_radius(nbdata-bkg, x, y, a, b, theta,6.0)
#     kron_rad[kron_rad<3.5/2.5] = 3.5/2.5
#     KRON = np.concatenate((KRON,kron_rad))
    
#     for s in filt:
#         d,r = Import(s)
#         bkg = sep.Background(d,bw=128,bh=128)
#         r = np.sqrt(r**2+bkg.rms()**2)
#         kron_rad[kron_rad<3.5/2.5] = 3.5/2.5
#         #f1,f1r,ff1 = sep.sum_ellipse(d-bkg, x, y, a,b,theta,2.5*kron_rad,err=r,subpix=1)
#         f1,f1r,ff1 = sep.sum_circle(d-bkg, x, y,2.5*kron_rad,err=r,subpix=1)
        
#         FLUX[s] = np.concatenate((FLUX[s],f1))
#         FLUXERR[s] = np.concatenate((FLUXERR[s],f1r))
#         FLAG[s] = np.concatenate((FLAG[s],ff1))

# for s in filt:
#     FLUXERR[s] *= 2.7


def saver(lib,name='FLUX',filt = ['Y','NB118','J','H','Ks']):
    x1 = len(lib)
    x2 = len(lib['NB118'])
    
    arr = np.zeros([x1,x2])
    for i,s in enumerate(filt):
        arr[i] = lib[s]
    np.save('.//SavedObjectsNpy/%s'%name,arr)
def loader(name,filt = ['Y','NB118','J','H','Ks']):
    arr = np.load('.//SavedObjectsNpy/%s.npy'%name)
    lib = {}
    for i,s in enumerate(filt):
        lib[s] = arr[i]
    return lib

# saver(FLUX,'FLUX')
# saver(FLUXERR,'FLUXERR')
# saver(FLAG,'FLAG')
# np.save('.//SavedObjectsNpy/%s'%'KRON',KRON)



FLUX = loader('FLUX')
FLUXERR = loader('FLUXERR')
FLAG = loader('FLAG')


kron_rad = np.load('.//SavedObjectsNpy/%s.npy'%'KRON')
kron_rad[kron_rad<0.3] = 0.3
objects = np.load('.//SavedObjectsNpy/Objects%i.npy'%0)
for i in range(len(pixxers)-1):
    OBS = np.load('.//SavedObjectsNpy/Objects%i.npy'%(i+1))
    objects = np.concatenate((objects,OBS))
a,b,theta = objects['a'], objects['b'], objects['theta']






hdu = fits.open(r'..\RawData\NB118.fits')[0]
w = wcs.WCS(hdu.header)
coord = np.array([Ra,Dec]).T
pix = np.round(w.wcs_world2pix(coord,True)).astype(int)
x,y = pix.T

plt.close('all')



plt.figure()
plt.hist2d(Ra,Dec,200)
plt.gca().invert_xaxis()
plt.colorbar()





def MAG(F,Fr):
    a = 5*np.sqrt(Fr**2/F**2)
    b = 2*np.log(2) + 2*np.log(5)
    return 30-2.5*np.log10(F),a/b

def MagToFlux(m,mr,filt):
    b = 5/2*np.log(10)
    filts = ['Y', 'NB118', 'J', 'H', 'Ks']
    cc = np.array([[24.97464949,  2.69066601],
       [25.08294148,  2.97287482],
       [25.78276679,  5.66378429],
       [25.35448342,  3.81762574],
       [25.4677102 ,  4.23724995]])
    ccr =np.array([[0.05065549, 0.1255342 ],
       [0.03595484, 0.09844899],
       [0.05337137, 0.27841382],
       [0.07023685, 0.24696384],
       [0.04977418, 0.19425132]])
    AB,a  =  cc [filts.index(filt)]
    ABr,ar = ccr[filts.index(filt)]
    
    F = np.exp(b*(AB-m))/a
    A = np.exp(2*b*(AB-m))
    B = b**2*ABr**2*a**2  +  b**2*mr**2*a**2  +  ar**2
    Fr = np.sqrt(A*B)/(a**4)
    return F,Fr
def ImgFluxToFlux(F,Fr,filt='NB118'):
    m,mr = MAG(F,Fr)
    return MagToFlux(m,mr,filt=filt)






flux, fluxerr, flag = FLUX['NB118'],FLUXERR['NB118'],FLAG['NB118']
flux,fluxerr = ImgFluxToFlux(flux,fluxerr)

sel1 = np.where( (flux >= 3*fluxerr) & (flux < 10000) & (fluxerr > 0) )[0]
def DoubletRemover(sel,Ra=Ra,Dec=Dec):
    sel2 = []
    sel2.append(sel[0])
    def dist(x1,x2,y1,y2):
        return np.sqrt((x1-x2)**2+(y1-y2)**2)
    print('Revoming doublets (%i)'%len(sel))
    for i in range(len(sel)-1):
        i += 1
        x1 = Ra[sel[i]]
        y1 = Dec[sel[i]]
        x2 = Ra[sel2]
        y2 = Dec[sel2]
        if np.min(dist(x1,x2,y1,y2)) > 20*RA_pix_scale:
            sel2.append(sel[i])
    print('Total of %i doubles removed' % (len(sel)-len(sel2)))
    return np.asarray(sel2)


Ra0 = np.copy(Ra)
Dec0 = np.copy(Dec)
sel2 = DoubletRemover(np.arange(len(Ra0)),Ra0,Dec0)
Ra1,Dec1 = Ra0[sel2],Dec0[sel2]

sel = DoubletRemover(sel1)



Ra,Dec = Ra[sel],Dec[sel]
a,b,theta =a[sel],b[sel],theta[sel]
x,y = x[sel],y[sel]
Npix = objects['npix'][sel]
kron_rad = kron_rad[sel]
errx2 = objects['errx2'][sel]
erry2 = objects['erry2'][sel]
errxy = objects['errxy'][sel]
for i,s in enumerate(filt):
    FLUX[s] = FLUX[s][sel]
    FLUXERR[s] = FLUXERR[s][sel]

flux, fluxerr, flag = FLUX['NB118'],FLUXERR['NB118'],FLAG['NB118']

def slope(x,x1,x2,y1,y2): #Find slope and displacement from two points
    a = (y2-y1) / (x2-x1) 
    b = (x2*y1-x1*y2) / (x2-x1)
    return a*x+b
def RaDeStr(Ra,Dec):
    x = [];y = []
    x.append(np.floor(Ra*24/360))
    remainder = (Ra*24/360-np.floor(Ra*24/360))*60
    x.append(np.floor(remainder))
    remainder = (remainder - np.floor(remainder))*60
    x.append(remainder)
    str1 = '%ih %im %.4gs' % (x[0],x[1],x[2])
    y.append(np.floor(Dec))
    remainder = (Dec - np.floor(Dec))*60
    y.append(np.floor(remainder))
    remainder = (remainder - np.floor(remainder))*60
    y.append(remainder)
    str2 = "%i°%i'%.4g\"" % (y[0],y[1],y[2])
    return str1,str2
#%%
plt.close(fig=2)

plt.figure(2,figsize=(10,4),dpi=150)
plt.subplot(1,2,1)
plt.hist2d(Ra1,Dec1,300,vmin=0)
plt.gca().invert_xaxis()
plt.xlabel('Ra')
plt.ylabel('Dec')
plt.title(r'$N\!B118$ without duplicates')
plt.colorbar()
plt.subplot(1,2,2)
plt.hist2d(Ra,Dec,300,vmin=0)
plt.gca().invert_xaxis()
plt.colorbar()
plt.xlabel('Ra')
plt.ylabel('Dec')
plt.title(r'3$\sigma$ detection in $N\!B118$')
plt.subplots_adjust(top=0.947,bottom=0.106,left=0.051,right=1.0,hspace=0.215,wspace=0.071)


#%%
import PIL
PIL.Image.MAX_IMAGE_PIXELS = None
def ThumbNail(Ra,Dec,objectwidth=10,w=50,Nticks = 7,check = False,pixcoord=False,addRa=[],addDec=[],addSize=[]): #w = extension to each side in pixels
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
        plt.figure(figsize=(10,6),dpi=100)
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
def Projection(RA,DEC,size=10,width=500,filt='NB118'):
    from astropy.wcs import wcs
    from astropy.nddata import Cutout2D
    data = fits.open(r'..\RawData\%s.fits'%filt)[0].data
    hdu = fits.open(r'..\RawData\%s.fits'%filt)[0]
    w = wcs.WCS(hdu.header)
    coord = np.array([[RA],[DEC]]).T
    pix = np.round(w.wcs_world2pix(coord,True)[0]).astype(int)
    hdu.header.update(Cutout2D(data, position=pix, size=width, wcs=w).wcs.to_header())
    return wcs.WCS(hdu.header)
def ThumbNail2(RA,DEC,size=10,width=500,addRA=[],addDEC=[],addSize=[],pixcoord=False,subplot=False,axes=[],sigma=0,filters=['Ks','NB118','J'],Q=10,stretch=25):
    IRAC_SCALE = 0.00016672186260247024  #Deg/pixel
    UVIS_SCALE = 4.166063682364829e-05   #Deg/pixel
    if 'ch1' in filters or 'ch2' in filters:
        width = np.int(np.ceil(width*UVIS_SCALE/IRAC_SCALE))
    if width > 4000:
        
        print('A window larger than 4000x4000 pixels is not recommended')
        cont = input('do you want to continue anyways? [Y/N]')
        if cont in ['N','n','no']:
            return
    import warnings
    warnings.filterwarnings("ignore")
    width //= 2
    from astropy.wcs import wcs
    from astropy.nddata import Cutout2D
    from scipy.ndimage import gaussian_filter
    
    
    
    
    
    #filters = ['H','NB118','J']
    #filters = ['Ks','H','NB118']
    data = {}
    for i in range(len(filters)):
        data[filters[i]] = fits.open(r'..\RawData\%s.fits'%filters[i])[0].data
    hdu = fits.open(r'..\RawData\%s.fits'%filters[i])[0]
    w = wcs.WCS(hdu.header)
    coord = np.array([[RA],[DEC]]).T
    pix = np.round(w.wcs_world2pix(coord,True)[0]).astype(int)
    if pixcoord:
        return pix
    
    
    R = Cutout2D(data[filters[0]], position=pix, size=width*2, wcs=w).data
    G = Cutout2D(data[filters[1]], position=pix, size=width*2, wcs=w).data
    B = Cutout2D(data[filters[2]], position=pix, size=width*2, wcs=w).data
    hdu.header.update(Cutout2D(data[filters[-1] ], position=pix, size=width, wcs=w).wcs.to_header())
    
    
    image = make_lupton_rgb(R,G,B,Q=Q,stretch=stretch)
    
    
    
    if not subplot:
        fig = plt.figure(figsize=(11,10),dpi=100)
        axes = fig.add_subplot(111,projection=wcs.WCS(hdu.header))
    if sigma > 0:
        for i in range(3):
            image[:,:,i] = gaussian_filter(image[:,:,i],sigma=sigma)
    plt.imshow(image,origin='lower')
    
    draw_circle = plt.Circle((width,width), radius=size,fill=False,color='white')
    axes.add_artist(draw_circle)
    
    if len(addRA)>0:
        longcoord = np.array([addRA,addDEC]).T
        longpix = np.round(w.wcs_world2pix(longcoord,True)).astype(int)
        if len(addSize) == 0:
            addSize = np.ones(len(addRA))*size
        if len(addRA) > 2: 
            for i in range(len(addRA)):
                draw_circle = plt.Circle( (longpix[i][0]-pix[0]+width,longpix[i][1]-pix[1]+width), radius=addSize[i],fill=False,color='r')
                axes.add_artist(draw_circle)
        else:
            i = 0
            draw_circle = plt.Circle( (longpix[i][0]-pix[0]+width,longpix[i][1]-pix[1]+width), radius=addSize[i],fill=False,color='r')
            axes.add_artist(draw_circle)
            i = 1
            draw_circle = plt.Circle( (longpix[i][0]-pix[0]+width,longpix[i][1]-pix[1]+width), radius=addSize[i],fill=False,color='r',ls='--')
            axes.add_artist(draw_circle)
    
    if not subplot:
        plt.xlabel('RA')
        plt.ylabel('Dec')
        plt.show()


plt.close(fig=3)
ThumbNail(150,2,0,90000,addRa=Ra,addDec=Dec,addSize=kron_rad)
# plt.savefig('../Thesis/fig/NBextractSmall.png',dpi=300)
#%%


def ImportClassic():
    filename = 'COSMOS2020_CLASSIC_v1.8.1_formatted.fits'
    with fits.open(filename) as f:
        data = f[1].data
        hdr = f[1].header
    
    filtnames = ['CFHT_ustar', 'CFHT_u', 'HSC_g', 'HSC_r', 'HSC_i', 'HSC_z', 'HSC_y', 'UVISTA_Y', 'UVISTA_J', 'UVISTA_H', 'UVISTA_Ks', 
                 'SC_IB427', 'SC_IB464', 'SC_IA484', 'SC_IB505', 'SC_IA527', 'SC_IB574', 'SC_IA624', 'SC_IA679', 'SC_IB709', 'SC_IA738', 
                 'SC_IA767', 'SC_IB827', 'SC_NB711', 'SC_NB816', 'UVISTA_NB118', 'SC_B', 'SC_V', 'SC_rp', 'SC_ip', 'SC_zpp', 'IRAC_CH1', 
                 'IRAC_CH2', 'GALEX_FUV', 'GALEX_NUV']
    AlambdaDivEBV = [4.674, 4.807, 3.69, 2.715, 2.0, 1.515, 1.298, 1.213, 0.874, 0.565, 0.365, 4.261, 3.844, 3.622, 3.425, 3.265, 2.938, 
                     2.694, 2.431, 2.29, 2.151, 1.997, 1.748, 2.268, 1.787, 0.946, 4.041, 3.128, 2.673, 2.003, 1.466, 0.163, 0.112, 8.31, 8.742]
    
    aperture = 2 # ["]
    offset = data['total_off'+str(aperture)]
    ebv    = data['EBV']
    names_noaper = ['IRAC_CH1', 'IRAC_CH2', 'GALEX_FUV', 'GALEX_NUV']
    
    for i,name in enumerate(filtnames):
        if name not in names_noaper:
            str_flux     = name+'_FLUX_APER'   +str(aperture)
            str_flux_err = name+'_FLUXERR_APER'+str(aperture)
            str_mag      = name+'_MAG_APER'    +str(aperture)
            str_mag_err  = name+'_MAGERR_APER' +str(aperture)
        else:
            str_flux     = name+'_FLUX'
            str_flux_err = name+'_FLUXERR'
            str_mag      = name+'_MAG'
            str_mag_err  = name+'_MAGERR'    
        flux     = data[str_flux]
        flux_err = data[str_flux_err]
        mag      = data[str_mag]
        mag_err  = data[str_mag_err]    
    
        # apply aperture-to-total offset
        if name not in names_noaper:
            idx = (flux>0)
            flux[idx]     *= 10**(-0.4*offset[idx])
            flux_err[idx] *= 10**(-0.4*offset[idx])
            mag[idx]      += offset[idx]    
    
        # correct for Milky Way attenuation
        idx = (flux>0)
        atten = 10**(0.4*AlambdaDivEBV[i]*ebv[idx])
        flux[idx]     *= atten
        flux_err[idx] *= atten
        mag[idx]      += -2.5*np.log10(atten)    
    
        data[str_flux]     = flux
        data[str_flux_err] = flux_err
        data[str_mag]      = mag
        data[str_mag_err]  = mag_err
    return data,hdr
if 'datahdr' not in locals():
    print("COSMOS CLASSIC data not loaded, loading...")
    data , datahdr = ImportClassic()
else:
    print("COSMOS CLASSIC data already loaded, skipping read")

def nearest_index(array, value): #Find index of element closest to given value
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def LocateSelf(Ra1,Dec1,RRa,RDec):
    def dist(RA,DEC):
        a = np.asarray(RA  - RRa )
        b = np.asarray(DEC - RDec)
        return a**2+b**2
    iout = []
    for i in range(len(Ra1)):
        iout.append(nearest_index(dist(Ra1[i],Dec1[i]), 0))
    return np.asarray(iout)
    

def Detected(i_c,i_s,r=3e-4):
    a = np.asarray(data['ALPHA_J2000'][i_c] - Ra[i_s])
    b = np.asarray(data['DELTA_J2000'][i_c] - Dec[i_s])
    d = np.sqrt( a**2 + b**2 )
    return d<r


SH = 0.01
Area = np.where(    (data['ALPHA_J2000'] >= np.min(Ra)-SH  ) & ( data['ALPHA_J2000'] <= np.max(Ra)+SH )
                 &  (data['DELTA_J2000'] >= np.min(Dec)-SH ) & ( data['DELTA_J2000'] <= np.max(Dec)+SH  )
                 )[0]

CRa = data['ALPHA_J2000'][Area]
CDec = data['DELTA_J2000'][Area]
Csize = data['FLUX_RADIUS'][Area]

def w2p(RR,DD):
    return wcs.WCS(hdu.header).world_to_pixel_values(RR,DD)


X,Y = w2p(CRa,CDec)
#plt.plot(X,Y,'or',alpha=0.25)


Closest = LocateSelf(Ra, Dec, CRa, CDec) #Index of CLassic where each position is nearest from self
Det =  []
Not =  []
DetArr = Detected(Area[Closest],np.arange(len(Ra)),5*a*RA_pix_scale*kron_rad)
for i in range(len(Closest)):
    if DetArr[i]:#
        Det.append(i)
    else:
        Not.append(i)
Csel = Closest[Det]
Msel = Det


flux_to_err,Not = sort([flux[Not]/fluxerr[Not],Not],direction='des')


print('%.4g%% Overlap with    CLASSIC' % ( (len(Det))/(len(x)) * 100 ) )
print('%.4g%% Not detected in CLASSIC' % ( (len(Not))/(len(x)) * 100 ) )

def AllDist(RA,DEC,ra,dec):
    a = np.asarray(RA  - ra )
    b = np.asarray(DEC - dec)
    return np.sqrt(a**2+b**2)
def corr(nb,j,y,nbr,jr,yr,err=False):

    out = j-nb + 0.07
    outerr = np.sqrt(jr**2 + nbr**2)
    
    cond1 = np.where( ( (y-j) <= 0.45) & (y>0) )
    cond2 = np.where( ( (y-j) >  0.45) & (y>0) )
    
    out[cond1] = (j[cond1] - nb[cond1]) + 0.34*(y[cond1]-j[cond1])
    out[cond2] = (j[cond2] - nb[cond2]) + 0.153
    outerr[cond1] = np.sqrt( 1089*(jr[cond1])**2 + 2500*(nbr[cond1])**2 + 289*(yr[cond1])**2 ) / 50
    if err:
        return out, outerr
    else:
        return out


def PlotEllipse(x,y,a,b,theta,cut,axis,col='green',alpha=0.4):
    scale=2.5*kron_rad[cut]
    try:
        for i,s in enumerate(cut):
            ellipse = Ellipse((x[cut][i],y[cut][i]),a[cut][i]*scale[i],b[cut][i]*scale[i],theta[cut][i]*180/np.pi,fill=False,ec=col,alpha=alpha)
            axis.add_artist(ellipse)
    except:
        ellipse = Ellipse((x[cut],y[cut]),a[cut]*scale,b[cut]*scale,theta[cut]*180/np.pi,fill=False,ec=col,alpha=alpha)
        axis.add_artist(ellipse)

Notold = np.copy(Not)

closest_classic_source = Area[LocateSelf(Ra[Not],Dec[Not],CRa,CDec)]
closest_flag = data['FLAG_COMBINED'][closest_classic_source]

def FlagRemover(sel):
    
    out = []
    for i in range(len(sel)):
        if closest_flag[i] < 1:
            out.append(sel[i])
    #waste, out = sort([flux[Not]/fluxerr[Not],out],direction='des')
    return out
#%%
Not = FlagRemover(Notold)
Not = np.delete(Not,np.arange(21))
Not = np.delete(Not,np.arange(4)+1)
Not = np.delete(Not,[2,6,29,37,49,57,80])
Not = Not[np.arange(36)]





Fdata = pd.read_csv('Filters.csv',sep=', ')
def W(filt):
    idx = np.where(filt == Fdata['Filter'])[0][0]
    return np.asarray(Fdata['Mean'][idx]),np.asarray(Fdata['FWHM'][idx])/2
wave = []
waver =[]
Flux = np.zeros([len(filt),len(Not)])
Fluxerr = np.zeros([len(filt),len(Not)])
mag = np.zeros([len(filt),len(Not)])
magerr = np.zeros([len(filt),len(Not)])

for i in range(len(filt)):
    w1,w2 = W(filt[i])
    wave.append(w1);waver.append(w2)
    Flux[i],Fluxerr[i] = ImgFluxToFlux(FLUX[filt[i]][Not] , FLUXERR[filt[i]][Not], filt[i])
    mag[i],magerr[i] = MAG(FLUX[filt[i]][Not] , FLUXERR[filt[i]][Not])
wave = np.asarray(wave)
waver = np.asarray(waver)



plt.close(fig=6)
fig3 = plt.figure(6,figsize=(7,7),dpi=85)
for i in range(len(Not)):
    ax = plt.subplot(*subidex(len(Not)), i+1)
    plt.errorbar(wave,mag[:,i],xerr=waver,yerr=magerr[:,i],fmt='.k',capsize=2,ms=2)
    for i2 in range(len(filt)):
        plt.annotate('%.6g'%(Flux[i2,i]/Fluxerr[i2,i]), xy=(wave[i2],mag[i2,i]-1.1*magerr[i2,i]), ha='center')
    plt.gca().invert_yaxis()
plt.tight_layout()

fig1 = plt.close(fig=7)
fig3 = plt.figure(7,figsize=(7,7),dpi=120)

plWidth = 40



coordCL = np.array([data['ALPHA_J2000'],data['DELTA_J2000']]).T

Np = len(Not)
for i in range(Np): #len(Not)
    hdu1 = fits.open(r'..\RawData\NB118.fits')[0]
    w1 = wcs.WCS(hdu1.header)
    
    coord1 = np.array([[Ra[Not[i]]],[Dec[Not[i]]]]).T
    pix1 = np.round(w1.wcs_world2pix(coord1,True)[0]).astype(int)
    pix2 = w1.wcs_world2pix(coordCL,True)
    x1 = pix2[:,0]-pix1[0]
    x2 = pix2[:,1]-pix1[1]
    
    sel = np.where( np.sqrt(x1**2+x2**2) <= plWidth )[0]
    print(len(sel))
    ax = plt.subplot(*subidex(Np), i+1,projection=Projection(coord1[0][0], coord1[0][1],size=0,width=plWidth))
    R1 = []; D1=[] ; S1 = []
    if len(sel) > 0:
        R1 = np.concatenate(([Ra[Not[i]]],data['ALPHA_J2000'][sel]))
        D1 = np.concatenate(([Dec[Not[i]]],data['DELTA_J2000'][sel]))
        S1 = np.concatenate(([0],data['KRON_RADIUS'][sel]))
    ThumbNail2(coord1[0][0], coord1[0][1],kron_rad[Not][i]*2.5*0,plWidth,subplot=True,axes=ax,addRA=R1,addDEC=D1,addSize=S1,filters=['Ks','NB118','Y'],Q=0.001,stretch=5)

    plt.annotate('%i(%.3g)'%(i,(Flux[1]/Fluxerr[1])[i]),(0,0),color='white')
    #PlotEllipse([plWidth/2]*len(a),[plWidth/2]*len(a),a,b,theta,Not[i],axis=ax,col='white',alpha=0.8)
    #plt.scatter(coordCL[:,0][sel],coordCL[:,0][sel],color='red')
    plt.axis('off')
plt.subplots_adjust(top=1.0,bottom=0.0,left=0.0,right=1.0,hspace=0.01,wspace=0.01)

#plt.savefig('BigPlots.jpg', dpi=430)

fig1 = plt.close(fig=8)
fig3 = plt.figure(8,figsize=(7,7),dpi=120)
for i in range(Np): #len(Not)
    hdu1 = fits.open(r'..\RawData\NB118.fits')[0]
    w1 = wcs.WCS(hdu1.header)
    
    coord1 = np.array([[Ra[Not[i]]],[Dec[Not[i]]]]).T
    pix1 = np.round(w1.wcs_world2pix(coord1,True)[0]).astype(int)
    pix2 = w1.wcs_world2pix(coordCL,True)
    x1 = pix2[:,0]-pix1[0]
    x2 = pix2[:,1]-pix1[1]
    
    sel = np.where( np.sqrt(x1**2+x2**2) <= plWidth )[0]
    #print(len(sel))
    ax = plt.subplot(*subidex(Np), i+1,projection=Projection(coord1[0][0], coord1[0][1],size=0,width=plWidth))
    R1 = []; D1=[] ; S1 = []
    if len(sel) > 0:
        R1 = np.concatenate(([Ra[Not[i]]],data['ALPHA_J2000'][sel]))
        D1 = np.concatenate(([Dec[Not[i]]],data['DELTA_J2000'][sel]))
        S1 = np.concatenate(([0],data['KRON_RADIUS'][sel]))
    ThumbNail2(coord1[0][0], coord1[0][1],kron_rad[Not][i]*2.5*0,plWidth,subplot=True,axes=ax,addRA=R1,addDEC=D1,addSize=S1,filters=['Ks','J','Y'],Q=0.001,stretch=5)
    
    
    plt.annotate('%i(%.3g)'%(i,(Flux[1]/Fluxerr[1])[i]),(0,0),color='white')
    #PlotEllipse([plWidth/2]*len(a),[plWidth/2]*len(a),a,b,theta,Not[i],axis=ax,col='white',alpha=0.8)
    #plt.scatter(coordCL[:,0][sel],coordCL[:,0][sel],color='red')
    plt.axis('off')
plt.subplots_adjust(top=1.0,bottom=0.0,left=0.0,right=1.0,hspace=0.01,wspace=0.01)


NB,NBr = corr(mag[1],mag[2],mag[0],magerr[1],magerr[2],magerr[0],True)
plt.close(fig=9)
fig3 = plt.figure(9,figsize=(10,7),dpi=85)
s1 = np.where( (NBr<10) & (NB>-0.1) )[0]
plt.errorbar(mag[1][s1],NB[s1],NBr[s1],magerr[1][s1],fmt='.k',capsize=2,ms=2)
plt.xlabel('NB118 [mag]')
plt.ylabel(r'$(J$-NB188$)_{corr}$ [mag]')
plt.tight_layout()


#%%
Not = FlagRemover(Notold)
Not = np.delete(Not,np.arange(21))
Not = np.delete(Not,np.arange(4)+1)
Not = np.delete(Not,[2,6,29,37,49,57,80])
Not = Not[np.arange(36)]
Not = Not[[0,3,4,9,11]]

Fdata = pd.read_csv('Filters.csv',sep=', ')
def W(filt):
    idx = np.where(filt == Fdata['Filter'])[0][0]
    return np.asarray(Fdata['Mean'][idx]),np.asarray(Fdata['FWHM'][idx])/2
wave = []
waver =[]
Flux = np.zeros([len(filt),len(Not)])
Fluxerr = np.zeros([len(filt),len(Not)])
mag = np.zeros([len(filt),len(Not)])
magerr = np.zeros([len(filt),len(Not)])

for i in range(len(filt)):
    w1,w2 = W(filt[i])
    wave.append(w1);waver.append(w2)
    Flux[i],Fluxerr[i] = ImgFluxToFlux(FLUX[filt[i]][Not] , FLUXERR[filt[i]][Not], filt[i])
    mag[i],magerr[i] = MAG(FLUX[filt[i]][Not] , FLUXERR[filt[i]][Not])
wave = np.asarray(wave)
waver = np.asarray(waver)



fig1 = plt.close(fig=10)
fig3 = plt.figure(10,figsize=(4,10),dpi=100)

plWidth = 40

Np = len(Not)
for i in range(Np): #len(Not)
    hdu1 = fits.open(r'..\RawData\NB118.fits')[0]
    w1 = wcs.WCS(hdu1.header)
    
    coord1 = np.array([[Ra[Not[i]]],[Dec[Not[i]]]]).T
    pix1 = np.round(w1.wcs_world2pix(coord1,True)[0]).astype(int)
    pix2 = w1.wcs_world2pix(coordCL,True)
    x1 = pix2[:,0]-pix1[0]
    x2 = pix2[:,1]-pix1[1]
    
    sel = np.where( np.sqrt(x1**2+x2**2) <= plWidth )[0]
    print(len(sel))
    
    ax = plt.subplot(Np,2, 1+2*i,projection=Projection(coord1[0][0], coord1[0][1],size=0,width=plWidth))
    R1 = []; D1=[] ; S1 = []
    if len(sel) > 0:
        R1 = np.concatenate(([Ra[Not[i]]],data['ALPHA_J2000'][sel]))
        D1 = np.concatenate(([Dec[Not[i]]],data['DELTA_J2000'][sel]))
        S1 = np.concatenate(([0],data['KRON_RADIUS'][sel]))
    ThumbNail2(coord1[0][0], coord1[0][1],kron_rad[Not][i]*2.5,plWidth,subplot=True,axes=ax,addRA=R1,addDEC=D1,addSize=S1,filters=['Ks','NB118','Y'],Q=0.001,stretch=7)
    plt.annotate(r'NB$_{%i}$'%(i+1),(0,0),color='white')
    #plt.annotate(r'%s,%s'%(RaDeStr(Ra[Not][i],Dec[Not][i])),(0,35),color='white')
    
    plt.axis('off')
    
    
    ax = plt.subplot(Np,2, 1+2*i+1,projection=Projection(coord1[0][0], coord1[0][1],size=0,width=plWidth))
    ThumbNail2(coord1[0][0], coord1[0][1],kron_rad[Not][i]*2.5,plWidth,subplot=True,axes=ax,addRA=R1,addDEC=D1,addSize=S1,filters=['Ks','J','Y'],Q=0.001,stretch=7)
    plt.annotate('%.3g'%((Flux[1]/Fluxerr[1])[i]),(20,0),color='white', ha='center')
    plt.annotate(r'%s  %s'%(RaDeStr(Ra[Not][i],Dec[Not][i])),(20,36),color='white', ha='center')
    plt.axis('off')
    


plt.subplots_adjust(top=1.0,bottom=0.0,left=0.0,right=1.0,hspace=0.01,wspace=0.01)




#%%

plt.close(fig=11)
fig3 = plt.figure(11,figsize=(3,3),dpi=120)

hsel = (kron_rad[Msel] > 3.5/2.5) & (data['KRON_RADIUS'][Csel]>3.5)

KRs = kron_rad[Msel][hsel]*2.5
KRc = data['KRON_RADIUS'][Csel][hsel]
sigma = (flux/fluxerr)[Msel][hsel]
# plt.subplot(1,2,1)
plt.hist(KRc,100,range=(3.5,15),fc='k',label=r'${ \tt Classic}$')
# plt.subplot(1,2,2)
plt.hist(KRs,100,range=(3.5,15),fc='g',label='NB118 extract',alpha=0.75)
plt.xlim([3.5,15])
plt.legend()
plt.xlabel('Kron Radius [pix]')
plt.ylabel('Number of sources')
plt.tight_layout()

plt.close(fig=12)
fig3 = plt.figure(12,figsize=(6,3),dpi=120)
sc = 1
ax = plt.axes([.08,.15,.38,.77])
plt.hist2d(sigma,KRc*sc,75,range=[[np.min(sigma),15],[3.5*sc,12*sc]],norm=LogNorm(vmin=1,vmax=30))
plt.ylabel('Kron Radius [pix]')
plt.xlabel('Significance [$\sigma$]')
plt.title(r'${ \tt Classic}$')

ax = plt.axes([.55,.15,.45,.77])
plt.hist2d(sigma,KRs*sc,75,range=[[np.min(sigma),15],[3.5*sc,12*sc]],norm=LogNorm(vmin=1,vmax=30))
plt.colorbar()
plt.ylabel('Kron Radius [pix]')
plt.xlabel('Significance [$\sigma$]')
plt.title('NB118 extract')



# plt.figure()
# plt.hist(data['KRON_RADIUS'][Csel],200)
# plt.close(fig=13)
# dr1 = [149.3399,149.3385,149.33930317102087]
# dc1 = [2.73843,2.73838,2.7384190324850715]
# sss = [1]*(len(dr1)-1)
# sss.append(1)

# ThumbNail2(149.33930317102087,2.7384190324850715,0,100,addRA=dr1,addDEC=dc1,addSize=sss)
# plt.tight_layout()









#%% 

x = objects['x']
y = objects['y']

Ra,Dec = wcs.WCS(hdu.header).pixel_to_world_values(x,y)
a,b,theta = objects['a'], objects['b'], objects['theta']
x,y,flag = sep.winpos(nbdata,x,y,3)

R = Cutout2D(fits.open(r'..\RawData\Ks.fits')[0].data, position=pix, size=width*2, wcs=w).data
G = Cutout2D(fits.open(r'..\RawData\NB118.fits')[0].data, position=pix, size=width*2, wcs=w).data
B = Cutout2D(fits.open(r'..\RawData\Y.fits')[0].data, position=pix, size=width*2, wcs=w).data
image = make_lupton_rgb(R,G,B,Q=0.1,stretch=10)




filt = ['Y','NB118','J','H','Ks']
FLUX = {}
FLUXERR = {}
FLAG = {}





kron_rad,kron_flag = sep.kron_radius(nbdata, x, y, a, b, theta,6.0)

for i in range(len(filt)):
    d,r = Import(filt[i])
    bkg = sep.Background(d,bw=128,bh=128)
    r = np.sqrt(r**2+bkg.rms()**2)
    #FLUX[filt[i]],FLUXERR[filt[i]],FLAG[filt[i]] = sep.sum_ellipse(d, x, y, a,b,theta,2.5*kron_rad,err=r,subpix=1)
    
    FLUX[filt[i]],FLUXERR[filt[i]],FLAG[filt[i]] = sep.sum_circann(d-bkg, x, y, RA_pix_scale*2)
    
    #FLUX[filt[i]],FLUXERR[filt[i]],FLAG[filt[i]] = sep.sum_ellipann(d-bkg, x, y, a,b,theta,2.5*kron_rad,2.5*kron_rad+0.3,err=r,subpix=1)

    FLUXERR[filt[i]] *= 2.7
    
#Error correction from cosmos2020


def MAG(F,Fr):
    a = 5*np.sqrt(Fr**2/F**2)
    b = 2*np.log(2) + 2*np.log(5)
    return 30-2.5*np.log10(F),a/b

def MagToFlux(m,mr,filt):
    b = 5/2*np.log(10)
    filts = ['Y', 'NB118', 'J', 'H', 'Ks']
    cc = np.array([[24.97464949,  2.69066601],
       [25.08294148,  2.97287482],
       [25.78276679,  5.66378429],
       [25.35448342,  3.81762574],
       [25.4677102 ,  4.23724995]])
    ccr =np.array([[0.05065549, 0.1255342 ],
       [0.03595484, 0.09844899],
       [0.05337137, 0.27841382],
       [0.07023685, 0.24696384],
       [0.04977418, 0.19425132]])
    AB,a  =  cc [filts.index(filt)]
    ABr,ar = ccr[filts.index(filt)]
    
    F = np.exp(b*(AB-m))/a
    A = np.exp(2*b*(AB-m))
    B = b**2*ABr**2*a**2  +  b**2*mr**2*a**2  +  ar**2
    Fr = np.sqrt(A*B)/(a**4)
    return F,Fr
def ImgFluxToFlux(F,Fr,filt='NB118'):
    m,mr = MAG(F,Fr)
    return MagToFlux(m,mr,filt=filt)


flux, fluxerr, flag = FLUX['NB118'],FLUXERR['NB118'],FLAG['NB118']
flux,fluxerr = ImgFluxToFlux(flux,fluxerr)
sel = np.where( (flux >= 5*fluxerr) & (flux < 500) & (fluxerr > 0) )[0]



Ra,Dec = Ra[sel],Dec[sel]
a,b,theta =a[sel],b[sel],theta[sel]
x,y = x[sel],y[sel]
Npix = objects['npix'][sel]
kron_rad,kron_flag = kron_rad[sel],kron_flag[sel]
for i,s in enumerate(filt):
    FLUX[s] = FLUX[s][sel]
    FLUXERR[s] = FLUXERR[s][sel]

flux, fluxerr, flag = FLUX['NB118'],FLUXERR['NB118'],FLAG['NB118']

#%%


plt.close(fig=3)
fig3 = plt.figure(3,figsize=(15,15),dpi=80)
ax = fig3.add_subplot(111,projection=wcs.WCS(hdu.header))
ax.imshow(image,origin='lower')
plt.xlabel('Ra')
plt.ylabel('Dec')
fig3.subplots_adjust(top=0.985,bottom=0.059,left=0.083,right=0.985,hspace=0.2,wspace=0.2)







def ImportClassic():
    filename = 'COSMOS2020_CLASSIC_v1.8.1_formatted.fits'
    with fits.open(filename) as f:
        data = f[1].data
        hdr = f[1].header
    
    filtnames = ['CFHT_ustar', 'CFHT_u', 'HSC_g', 'HSC_r', 'HSC_i', 'HSC_z', 'HSC_y', 'UVISTA_Y', 'UVISTA_J', 'UVISTA_H', 'UVISTA_Ks', 
                 'SC_IB427', 'SC_IB464', 'SC_IA484', 'SC_IB505', 'SC_IA527', 'SC_IB574', 'SC_IA624', 'SC_IA679', 'SC_IB709', 'SC_IA738', 
                 'SC_IA767', 'SC_IB827', 'SC_NB711', 'SC_NB816', 'UVISTA_NB118', 'SC_B', 'SC_V', 'SC_rp', 'SC_ip', 'SC_zpp', 'IRAC_CH1', 
                 'IRAC_CH2', 'GALEX_FUV', 'GALEX_NUV']
    AlambdaDivEBV = [4.674, 4.807, 3.69, 2.715, 2.0, 1.515, 1.298, 1.213, 0.874, 0.565, 0.365, 4.261, 3.844, 3.622, 3.425, 3.265, 2.938, 
                     2.694, 2.431, 2.29, 2.151, 1.997, 1.748, 2.268, 1.787, 0.946, 4.041, 3.128, 2.673, 2.003, 1.466, 0.163, 0.112, 8.31, 8.742]
    
    aperture = 2 # ["]
    offset = data['total_off'+str(aperture)]
    ebv    = data['EBV']
    names_noaper = ['IRAC_CH1', 'IRAC_CH2', 'GALEX_FUV', 'GALEX_NUV']
    
    for i,name in enumerate(filtnames):
        if name not in names_noaper:
            str_flux     = name+'_FLUX_APER'   +str(aperture)
            str_flux_err = name+'_FLUXERR_APER'+str(aperture)
            str_mag      = name+'_MAG_APER'    +str(aperture)
            str_mag_err  = name+'_MAGERR_APER' +str(aperture)
        else:
            str_flux     = name+'_FLUX'
            str_flux_err = name+'_FLUXERR'
            str_mag      = name+'_MAG'
            str_mag_err  = name+'_MAGERR'    
        flux     = data[str_flux]
        flux_err = data[str_flux_err]
        mag      = data[str_mag]
        mag_err  = data[str_mag_err]    
    
        # apply aperture-to-total offset
        if name not in names_noaper:
            idx = (flux>0)
            flux[idx]     *= 10**(-0.4*offset[idx])
            flux_err[idx] *= 10**(-0.4*offset[idx])
            mag[idx]      += offset[idx]    
    
        # correct for Milky Way attenuation
        idx = (flux>0)
        atten = 10**(0.4*AlambdaDivEBV[i]*ebv[idx])
        flux[idx]     *= atten
        flux_err[idx] *= atten
        mag[idx]      += -2.5*np.log10(atten)    
    
        data[str_flux]     = flux
        data[str_flux_err] = flux_err
        data[str_mag]      = mag
        data[str_mag_err]  = mag_err
    return data,hdr
if 'datahdr' not in locals():
    print("COSMOS CLASSIC data not loaded, loading...")
    data , datahdr = ImportClassic()
else:
    print("COSMOS CLASSIC data already loaded, skipping read")

def nearest_index(array, value): #Find index of element closest to given value
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def LocateSelf(Ra1,Dec1,RRa,RDec):
    def dist(RA,DEC):
        a = np.asarray(RA  - RRa )
        b = np.asarray(DEC - RDec)
        return np.sqrt(a**2+b**2)
    iout = []
    for i in range(len(Ra1)):
        iout.append(nearest_index(dist(Ra1[i],Dec1[i]), 0))
    return np.asarray(iout)
    

def Detected(i_c,i_s,r=3e-4):
    a = np.asarray(data['ALPHA_J2000'][i_c] - Ra[i_s])
    b = np.asarray(data['DELTA_J2000'][i_c] - Dec[i_s])
    d = np.sqrt( a**2 + b**2 )
    return d<r

coordsel = wcs.WCS(hdu.header).pixel_to_world_values([0,2*width],[0,2*width])

Area = np.where(    (data['ALPHA_J2000'] >= np.min(coordsel[0])  ) & ( data['ALPHA_J2000'] <= np.max(coordsel[0])  )
                 &  (data['DELTA_J2000'] >= np.min(coordsel[1]) ) & ( data['DELTA_J2000'] <= np.max(coordsel[1])  )
                 )[0]

CRa = data['ALPHA_J2000'][Area]
CDec = data['DELTA_J2000'][Area]
Csize = data['FLUX_RADIUS'][Area]

def w2p(RR,DD):
    return wcs.WCS(hdu.header).world_to_pixel_values(RR,DD)


X,Y = w2p(CRa,CDec)
plt.plot(X,Y,'or',alpha=0.25)


Closest = LocateSelf(Ra, Dec, CRa, CDec) #Index of CLassic where each position is nearest from self
Det =  []
Not =  []
for i in range(len(Closest)):
    if Detected(Area[Closest][i],i,5*a[i]*RA_pix_scale*kron_rad[i]):#
        Det.append(i)
    else:
        Not.append(i)
Csel = Closest[Det]
Msel = Det


flux_to_err,Not = sort([flux[Not]/fluxerr[Not],Not],direction='des')


print('%.4g%% Overlap with    CLASSIC' % ( (len(Det))/(len(x)) * 100 ) )
print('%.4g%% Not detected in CLASSIC' % ( (len(Not))/(len(x)) * 100 ) )

def AllDist(RA,DEC,ra,dec):
    a = np.asarray(RA  - ra )
    b = np.asarray(DEC - dec)
    return np.sqrt(a**2+b**2)





#%%
def PlotEllipse(x,y,a,b,theta,cut,axis,col='green',alpha=0.4):
    scale=2.5*kron_rad[cut]
    for i in range(len(cut)):
        ellipse = Ellipse((x[cut][i],y[cut][i]),a[cut][i]*scale[i],b[cut][i]*scale[i],theta[cut][i]*180/np.pi,fill=False,ec=col,alpha=alpha)
        axis.add_artist(ellipse)

imageNB  = make_lupton_rgb(G,G,G,Q=0.001,stretch=5)
plt.close(fig=4)
fig3 = plt.figure(4,figsize=(15,15),dpi=80)
ax = fig3.add_subplot(111,projection=wcs.WCS(hdu.header))
ax.imshow(image,origin='lower')
plt.xlabel('Ra')
plt.ylabel('Dec')
plt.title('NB118 only')
fig3.subplots_adjust(top=0.985,bottom=0.059,left=0.083,right=0.985,hspace=0.2,wspace=0.2)
plt.scatter(X[Csel],Y[Csel],fc='r',alpha=0.25)
plt.scatter(x[Msel],y[Msel],fc='g')
PlotEllipse(x,y,a,b,theta,Msel,ax)


plt.close(fig=5)
fig3 = plt.figure(5,figsize=(15,15),dpi=80)
ax = fig3.add_subplot(111,projection=wcs.WCS(hdu.header))
ax.imshow(image,cmap='binary')
plt.xlabel('Ra')
plt.ylabel('Dec')
plt.title('NB118 only')
for i in range(len(Not)):
    plt.annotate('%i'%i,(x[Not[i]],y[Not[i]]),color='white',zorder=1000)
fig3.subplots_adjust(top=0.985,bottom=0.059,left=0.083,right=0.985,hspace=0.2,wspace=0.2)



PlotEllipse(x,y,a,b,theta,Not,ax,alpha=1)
plt.scatter(x[Not],y[Not],fc='g')
plt.scatter(X,Y,fc='r',alpha=0.5)

#%%
Fdata = pd.read_csv('Filters.csv',sep=', ')
def W(filt):
    idx = np.where(filt == Fdata['Filter'])[0][0]
    return np.asarray(Fdata['Mean'][idx]),np.asarray(Fdata['FWHM'][idx])/2
wave = []
waver =[]
Flux = np.zeros([len(filt),len(Not)])
Fluxerr = np.zeros([len(filt),len(Not)])
mag = np.zeros([len(filt),len(Not)])
magerr = np.zeros([len(filt),len(Not)])

for i in range(len(filt)):
    w1,w2 = W(filt[i])
    wave.append(w1);waver.append(w2)
    Flux[i],Fluxerr[i] = ImgFluxToFlux(FLUX[filt[i]][Not] , FLUXERR[filt[i]][Not], filt[i])
    mag[i],magerr[i] = MAG(FLUX[filt[i]][Not] , FLUXERR[filt[i]][Not])
wave = np.asarray(wave)
waver = np.asarray(waver)

def corr(nb,j,y,nbr,jr,yr,err=False):

    out = j-nb + 0.07
    outerr = np.sqrt(jr**2 + nbr**2)
    
    cond1 = np.where( ( (y-j) <= 0.45) & (y>0) )
    cond2 = np.where( ( (y-j) >  0.45) & (y>0) )
    
    out[cond1] = (j[cond1] - nb[cond1]) + 0.34*(y[cond1]-j[cond1])
    out[cond2] = (j[cond2] - nb[cond2]) + 0.153
    outerr[cond1] = np.sqrt( 1089*(jr[cond1])**2 + 2500*(nbr[cond1])**2 + 289*(yr[cond1])**2 ) / 50
    if err:
        return out, outerr
    else:
        return out






# mag,magerr = MAG(Flux,Fluxerr)


plt.close(fig=6)
fig3 = plt.figure(6,figsize=(15,15),dpi=85)
for i in range(len(Not)):
    ax = plt.subplot(*subidex(len(Not)), i+1)
    plt.errorbar(wave,mag[:,i],xerr=waver,yerr=magerr[:,i],fmt='.k',capsize=2,ms=2)
    for i2 in range(len(filt)):
        plt.annotate('%.6g'%(Flux[i2,i]/Fluxerr[i2,i]), xy=(wave[i2],mag[i2,i]-1.1*magerr[i2,i]), ha='center')
    plt.gca().invert_yaxis()
plt.tight_layout()

plt.close(fig=7)
fig3 = plt.figure(7,figsize=(15,15),dpi=85)

plWidth = 25

for i in range(len(Not)):
    ax = plt.subplot(*subidex(len(Not)), i+1,projection=wcs.WCS(hdu.header))
    
    hdu1 = fits.open(r'..\RawData\NB118.fits')[0]
    w1 = wcs.WCS(hdu1.header)
    coord1 = np.array([[Ra[Not[i]]],[Dec[Not[i]]]]).T
    pix1 = np.round(w1.wcs_world2pix(coord1,True)[0]).astype(int)
    img1 = Cutout2D(fits.open(r'..\RawData\%s.fits'%'Ks')[0].data, position=pix1, size=plWidth*2, wcs=w1).data
    img2 = Cutout2D(fits.open(r'..\RawData\%s.fits'%'NB118')[0].data, position=pix1, size=plWidth*2, wcs=w1).data
    img3 = Cutout2D(fits.open(r'..\RawData\%s.fits'%'Y')[0].data, position=pix1, size=plWidth*2, wcs=w1).data
    plt.imshow(make_lupton_rgb(img2,img2,img2,Q=0.001,stretch=5))
    plt.annotate('%i'%i,(plWidth,plWidth),color='white')
    plt.axis('off')
plt.tight_layout()




NB,NBr = corr(mag[1],mag[2],mag[0],magerr[1],magerr[2],magerr[0],True)
plt.close(fig=8)
fig3 = plt.figure(8,figsize=(10,7),dpi=85)

plt.errorbar(mag[1],NB,NBr,magerr[1],fmt='.k',capsize=2,ms=2)
plt.xlabel('NB118 [mag]')
plt.ylabel(r'$(J$-NB188$)_{corr}$ [mag]')
plt.tight_layout()

#%%
# plt.close(fig=8)
# fig3 = plt.figure(8,figsize=(15,15),dpi=85)
# relsel = np.where( (data['UVISTA_J_FLUX_APER2']>= 0) & (data['UVISTA_J_FLUX_APER2']< 40) )[0]
# plt.scatter(data['UVISTA_J_FLUX_APER2'][relsel],data['UVISTA_J_MAG_APER2'][relsel])
# from scipy.optimize import curve_fit
# def fit(x,AB,a):
#     return AB - 5/2*np.log10(x*a)
# arg = np.linspace(1e-5,40,500)
# for s in filt[2]:
#     ab,cov = curve_fit(fit,data['UVISTA_%s_FLUX_APER2'%s][relsel],data['UVISTA_%s_MAG_APER2'%s][relsel],p0=(21,1/10))
# plt.plot(arg,fit(arg,*ab),'k')


# def MF(m,AB,a):
#     A = 2/5 * np.log(10) * ( AB - m )
#     return np.exp(A)/a


# plt.close(fig=9)
# fig3 = plt.figure(9,figsize=(15,15),dpi=85)

# s = 'Y'
# relsel = np.where( (data['UVISTA_%s_FLUX_APER2'%s]>= 0) & (data['UVISTA_%s_FLUX_APER2'%s]< 1e+10) )[0]
# plt.scatter(data['UVISTA_%s_MAG_APER2'%s][relsel],data['UVISTA_%s_FLUX_APER2'%s][relsel])
# ab,cov = curve_fit(MF,data['UVISTA_%s_MAG_APER2'%s][relsel],data['UVISTA_%s_FLUX_APER2'%s][relsel],p0=(25,3))
# mm = np.linspace(10,40,500)
# plt.plot(mm,MF(mm,*ab),'-k')


# coefs = np.zeros([len(filt),2])
# coefsr = np.zeros([len(filt),2])
# for i,s in enumerate(filt):
#     relsel = np.where( (data['UVISTA_%s_FLUX_APER2'%s]>= 0) & (data['UVISTA_%s_FLUX_APER2'%s]< 1e+10) )[0]
#     ab,cov = curve_fit(MF,data['UVISTA_%s_MAG_APER2'%s][relsel],data['UVISTA_%s_FLUX_APER2'%s][relsel]
#                        ,sigma=data['UVISTA_%s_FLUXERR_APER2'%s][relsel],p0=(25,4))
#     coefs[i] = ab
#     coefsr[i] = np.sqrt(np.abs(np.diag(cov)))




# def MagToFlux(m,mr,filt):
#     b = 5/2*np.log(10)
#     filts = ['Y', 'NB118', 'J', 'H', 'Ks']
#     cc = np.array([[24.97464949,  2.69066601],
#        [25.08294148,  2.97287482],
#        [25.78276679,  5.66378429],
#        [25.35448342,  3.81762574],
#        [25.4677102 ,  4.23724995]])
#     ccr =np.array([[0.05065549, 0.1255342 ],
#        [0.03595484, 0.09844899],
#        [0.05337137, 0.27841382],
#        [0.07023685, 0.24696384],
#        [0.04977418, 0.19425132]])
#     AB,a  =  cc [filts.index(filt)]
#     ABr,ar = ccr[filts.index(filt)]
    
#     F = np.exp(b*(AB-m))/a
#     A = np.exp(2*b*(AB-m))
#     B = b**2*ABr**2*a**2  +  b**2*mr**2*a**2  +  ar**2
#     Fr = np.sqrt(A*B)/(a**4)
#     return F,Fr













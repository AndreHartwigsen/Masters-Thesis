#%%    Loading data and functions
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
from astropy.io import fits
import pandas as pd
from scipy.ndimage import gaussian_filter
from astropy.visualization import make_lupton_rgb
from scipy.interpolate import interp1d
# Jy = 1e-26
# mJy = 1e-29



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
    return str1,str2#,'%.2i:%.2i:%.3g %.1i:%.2i:%.4g'% (x[0],x[1],x[2],y[0],y[1],y[2])
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
    
    draw_circle = plt.Circle((width,width), radius=size,fill=False,color='r')
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
            draw_circle = plt.Circle( (longpix[i][0]-pix[0]+width,longpix[i][1]-pix[1]+width), radius=addSize[i],fill=False,color='b',ls='--')
            axes.add_artist(draw_circle)
    
    if not subplot:
        plt.xlabel('RA')
        plt.ylabel('Dec')
        plt.show()
    # else:
    #     plt.axis('off')


def nearest_index(array, value): #Find index of element closest to given value
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
def zTarget(target,sigma_target=None,cen=11850,sigma_cen=20): #All in angstrom
    nb = cen
    snb = sigma_cen
    if sigma_target == None:
        st = 0
    else:
        st = sigma_target
    z = nb/target - 1
    sz = np.sqrt( (st**2 * nb**2 + snb**2 * target**2) / (target**4) )
    return z,sz
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
def ImportFarmer():
    filename = 'COSMOS2020_FARMER_v1.8.1_formatted.fits'
    with fits.open(filename) as f:
        data = f[1].data
        hdr = f[1].header
    
    filtnames = ['CFHT_ustar', 'CFHT_u', 'HSC_g', 'HSC_r', 'HSC_i', 'HSC_z', 'HSC_y', 'UVISTA_Y', 'UVISTA_J', 'UVISTA_H', 'UVISTA_Ks', 
                 'SC_IB427', 'SC_IB464', 'SC_IA484', 'SC_IB505', 'SC_IA527', 'SC_IB574', 'SC_IA624', 'SC_IA679', 'SC_IB709', 'SC_IA738', 
                 'SC_IA767', 'SC_IB827', 'SC_NB711', 'SC_NB816', 'UVISTA_NB118', 'SC_B', 'SC_V', 'SC_rp', 'SC_ip', 'SC_zpp', 'IRAC_CH1', 
                 'IRAC_CH2', 'GALEX_FUV', 'GALEX_NUV']
    AlambdaDivEBV = [4.674, 4.807, 3.69, 2.715, 2.0, 1.515, 1.298, 1.213, 0.874, 0.565, 0.365, 4.261, 3.844, 3.622, 3.425, 3.265, 2.938, 
                     2.694, 2.431, 2.29, 2.151, 1.997, 1.748, 2.268, 1.787, 0.946, 4.041, 3.128, 2.673, 2.003, 1.466, 0.163, 0.112, 8.31, 8.742]
    
    #aperture = 2 # ["]
    #offset = data['total_off'+str(aperture)]
    ebv    = data['EBV']
    names_noaper = ['UVISTA_NB118', 'SC_B', 'SC_V', 'SC_rp', 'SC_ip', 'SC_zpp', 'IRAC_CH1', 'IRAC_CH2']
    
    for i,name in enumerate(filtnames):
        if name not in names_noaper:
            str_flux     = name+'_FLUX'   
            str_flux_err = name+'_FLUXERR'
            str_mag      = name+'_MAG'    
            str_mag_err  = name+'_MAGERR' 
    
        flux     = data[str_flux]
        flux_err = data[str_flux_err]
        mag      = data[str_mag]
        mag_err  = data[str_mag_err]    
    
        # apply aperture-to-total offset
        # if name not in names_noaper:
        #     idx = (flux>0)
        #     flux[idx]     *= 10**(-0.4*offset[idx])
        #     flux_err[idx] *= 10**(-0.4*offset[idx])
        #     mag[idx]      += offset[idx]    
    
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

#                Lalpha       OII     Hdelta  Hgamma  Hbeta   OIII    OIII    Halpha
# emit = np.array([1215.022732, 3727.3, 4102.8, 4340.0, 4861.3, 4959.0, 5006.8, 6562.8])
# redshifts,sigma_redshifts = zTarget(emit)


### ----- LOADING catalogs ----- ###
if 'hdr' not in locals():
    print("COSMOS CLASSIC data not loaded, loading...")
    data , hdr = ImportClassic()
else:
    print("COSMOS CLASSIC data already loaded, skipping read")

if 'hdrF' not in locals():
    print("COSMOS FARMER data not loaded, loading...")
    # hduz = fits.open('COSMOS2020_FARMER_v1.8.1_formatted.fits')
    # hdrF = hduz[1].header
    # dataF = hduz[1].data
    dataF , hdrF = ImportFarmer()
else:
    print("COSMOS FARMER data already loaded, skipping read")


### ----- LOADING redshifts ----- ###
if 'zphot' not in locals():
    print("lephare_CLASSIC data not loaded, loading...")
    hduz = fits.open('photoz_cosmos2020_lephare_classic_v1.8.out.fits', ignore_missing_end=True)
    hdrz = hduz[1].header
    zphot = hduz[1].data
else:
    print("lephare_CLASSIC data already loaded, skipping read")
if 'zphot2' not in locals():
    print("Eazy_CLASSIC data not loaded, loading...")
    hduz = fits.open('EAZY_classic_v1.5.full.zout.fits', ignore_missing_end=True)
    hdrz2 = hduz[1].header
    zphot2 = hduz[1].data
else:
    print("Eazy_CLASSIC data already loaded, skipping read")
    
if 'zphotF' not in locals():
    print("lephare_FARMER data not loaded, loading...")
    hduz = fits.open('photoz_cosmos2020_lephare_farmer_v1.8.out.fits', ignore_missing_end=True)
    hdrzF = hduz[1].header
    zphotF = hduz[1].data
else:
    print("lephare_FARMER data already loaded, skipping read")
if 'zphot2F' not in locals():
    print("Eazy_FARMER data not loaded, loading...")
    hduz = fits.open('farmer_v051120.full.zout.fits', ignore_missing_end=True)
    hdrz2F = hduz[1].header
    zphot2F = hduz[1].data
    zphot2F = zphot2F[np.intersect1d(zphotF['Id'],zphot2F['id'],return_indices=True)[-1]]
else:
    print("Eazy_FARMER data already loaded, skipping read")



def xysize(cut):   #Determine the radius in pixels around the central pixel that detection was performed
    # a = 40000**2/(17940*11408)
    # npix = data['FLUX_RADIUS'][cut]
    # return np.sqrt(npix/(np.pi*a))*15
    return data['FLUX_RADIUS'][cut]*2
#                Lalpha       OII     Hdelta  Hgamma  Hbeta   OIII    OIII    Halpha
emit = np.array([1215.022732, 3727.3, 4102.8, 4340.0, 4861.3, 4959.0, 5006.8, 6562.8])
redshifts,sigma_redshifts = zTarget(emit)


aper = '_APER2'


def depth(filt,sigma=3,aper=aper):
    def mags(filt,sigma=sigma):
        where = np.where( (data['%s_FLUX'%filt +aper] >= sigma*data['%s_FLUXERR'%filt +aper]) & (data['%s_FLUX'%filt +aper] >= 0))[0]
        return data['%s_MAG'%filt +aper][where]
    if isinstance(filt,str):
        return np.max(mags(filt))
    elif isinstance(filt,list) or isinstance(filt,np.ndarray):
        out = []
        if isinstance(sigma,int) or isinstance(sigma,float):
            S = np.ones(len(filt))*sigma
        else:
            S = sigma
        for i in range(len(filt)):
            out.append(np.max(mags(filt[i],S[i])))
        return out
    else: 
        raise Warning('invalid input')

def spec(ii,err=False):
    add = ''
    if err:
        add = 'ERR'
    typesel='FLUX%s%s'% (add,aper)
    NB118 = data['UVISTA_NB118_'+typesel][ii]
    Y     = data['UVISTA_Y_'    +typesel][ii]
    J     = data['UVISTA_J_'    +typesel][ii]
    H     = data['UVISTA_H_'    +typesel][ii]
    Ks    = data['UVISTA_Ks_'   +typesel][ii]
    B     = data['SC_B_' +typesel][ii]
    V     = data['SC_V_' +typesel][ii]
    z     = data['HSC_z_' +typesel][ii]
    return np.array([B, V, z,Y, J, NB118, H, Ks])


def Data(sel,typesel='MAG'+aper):
    cc = 0
    if  'MAG' in typesel:
        cc = 1
    global NB118,Y,Z,J,H,Ks,B,V,g,r,y,z,i
    NB118 = data['UVISTA_NB118_'+typesel][sel]+0.85*cc
    Y     = data['UVISTA_Y_'    +typesel][sel]+0.61*cc
    J     = data['UVISTA_J_'    +typesel][sel]+0.92*cc
    H     = data['UVISTA_H_'    +typesel][sel]+1.38*cc
    Ks    = data['UVISTA_Ks_'   +typesel][sel]+1.84*cc
    B     = data['SC_B_' +typesel][sel]
    V     = data['SC_V_' +typesel][sel]
    g     = data['HSC_g_' +typesel][sel]
    r     = data['HSC_r_' +typesel][sel]
    i     = data['HSC_i_' +typesel][sel]
    z     = data['HSC_z_' +typesel][sel]
    y     = data['HSC_y_' +typesel][sel]


def LocateFarmer(Ra,Dec,FarmerCut=[]):
    if len(FarmerCut) == 0:
        farmer_ra = np.asarray(dataF['ALPHA_J2000'])
        farmer_dec= np.asarray(dataF['DELTA_J2000'])
    else:
        farmer_ra = np.asarray(dataF['ALPHA_J2000'])[FarmerCut]
        farmer_dec= np.asarray(dataF['DELTA_J2000'])[FarmerCut]
    def dist(RA,DEC):
        a = np.asarray(RA  - farmer_ra )
        b = np.asarray(DEC - farmer_dec)
        return np.sqrt(a**2+b**2)
    
    if isinstance(Ra,float):
        return nearest_index(dist(Ra,Dec), 0)
    else:
        iout = []
        for i in range(len(Ra)):
            iout.append(nearest_index(dist(Ra[i],Dec[i]), 0))
        return iout

def detected_in_farmer(i_c,i_f):
    a = np.asarray(data['ALPHA_J2000'][i_c] - dataF['ALPHA_J2000'][i_f])
    b = np.asarray(data['DELTA_J2000'][i_c] - dataF['DELTA_J2000'][i_f])
    d = np.sqrt(a**2+b**2)
    return d<1e-4

def corr(cut,err=False):
    nb = data['UVISTA_NB118_MAG'+aper][cut]
    j  = data['UVISTA_J_MAG'+aper][cut]
    y  = data['UVISTA_Y_MAG'+aper][cut]
    nbr = data['UVISTA_NB118_MAGERR'+aper][cut]
    jr  = data['UVISTA_J_MAGERR'+aper][cut]
    yr  = data['UVISTA_Y_MAGERR'+aper][cut]
    
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

def wmean(x,sx,axis=None):
    w = 1/sx**2
    m = np.sum(x*w,axis=axis)/np.sum(w,axis=axis)
    sm = 1/(np.sqrt(np.sum(w,axis=axis)))
    return m,sm



#%%            Data selection
plt.close('all')

SEDSetting = 1

if SEDSetting == 0:
    ### Selecting hoe much larger NB118 flux has to be than lower wavelength fluxes (Z, V, B, J and Y)
    order = 8
    
    FLUXCUT = data['UVISTA_NB118_FLUX'+aper]/order
    
    ranged1 = np.where( (FLUXCUT > data['HSC_g_FLUX'+aper    ]) 
                      & (FLUXCUT > data['HSC_r_FLUX'+aper    ])
                      & (FLUXCUT > data['HSC_i_FLUX'+aper    ])
                      & (FLUXCUT > data['HSC_z_FLUX'+aper    ])
                      & (FLUXCUT > data['HSC_y_FLUX'+aper    ])
                      & (FLUXCUT > data['SC_gp_FLUX'+aper    ]) 
                      & (FLUXCUT > data['SC_rp_FLUX'+aper    ])
                      & (FLUXCUT > data['SC_ip_FLUX'+aper    ])
                      & (FLUXCUT > data['SC_zp_FLUX'+aper    ])
                      & (FLUXCUT > data['UVISTA_Y_FLUX'+aper ])
                      & (zphot2['z_spec']<0) & (zphot['type'] == 0)
                      & (data['FLAG_SUPCAM'] == 0) & (data['FLAG_HSC'] == 0)
                      & (data['UVISTA_NB118_FLAGS'] == 0)
                      & (data['UVISTA_J_FLAGS'] == 0)
                     )[0]
    
    ranged2 = np.where( (data['UVISTA_Ks_MAGERR'+aper]        <= 10) & (data['UVISTA_Ks_MAG'+aper]         >= 0)
                      & (data['UVISTA_J_MAGERR'+aper ]        <= 10) & (data['UVISTA_J_MAG'+aper ]         >= 0)
                      & (data['UVISTA_H_MAGERR'+aper ]        <= 10) & (data['UVISTA_H_MAG'+aper ]         >= 0)
                      & (data['UVISTA_NB118_MAGERR'+aper ]    <= 10) & (data['UVISTA_NB118_MAG'+aper ]     >= 0)
                      )[0]
    
    
    ranged = np.intersect1d(ranged1,ranged2)
    
    
    JNB, JNBr = corr(ranged,True)
    ranged = ranged[ (JNB > 5/2*JNBr) & (JNB>1/5)]
    JNB, JNBr = corr(ranged,True)
    
    
    
    Fsel = ['HSC_g','HSC_r','HSC_i','HSC_z','HSC_y','SC_gp','SC_rp','SC_ip','SC_zp','UVISTA_Y']
    NB118 = data['UVISTA_NB118_FLUX'+aper][ranged]
    
    
    MINSEL = ['J','H','Ks','NB118']
    FLUXMIN = np.zeros([len(MINSEL),len(ranged)])
    for i in range(len(MINSEL)):
        FLUXMIN[i] = data['UVISTA_%s_FLUX%s'   %(MINSEL[i],aper)][ranged]
    FLUXMIN = np.min(FLUXMIN,axis=0)
    
    
    #Cut for narrowband
    cutn = np.where( data['UVISTA_J_FLUX'+aper][ranged] >= 3*data['UVISTA_J_FLUXERR'+aper][ranged])[0]
    cutn = np.intersect1d(cutn,np.where(NB118 >= 5*data['UVISTA_NB118_FLUXERR'+aper][ranged]  )[0])
    cutn = np.intersect1d(cutn,np.where(data['UVISTA_NB118_MAG'+aper][ranged] >= 15)[0])
    
    
    for i in range(len(Fsel)):
        FLUX    = data['%s_FLUX%s'   %(Fsel[i],aper)][ranged]
        FLUXERR = data['%s_FLUXERR%s'%(Fsel[i],aper)][ranged]
        cutc = np.where( (FLUX*order < NB118) & (FLUX < FLUXMIN))
        #cutc = np.where( (FLUX - 3*FLUXERR <= 0))
        
        cutn = np.intersect1d(cutn,cutc)
    cutn = np.delete(cutn,2)
if SEDSetting == 1:
    ### Selecting hoe much larger J flux has to be than lower wavelength fluxes (Z, V, B, J and Y)
    order = 8
    
    FLUXCUT = data['UVISTA_J_FLUX'+aper]/order
    
    ranged1 = np.where( (FLUXCUT > data['HSC_g_FLUX'+aper    ]) 
                      & (FLUXCUT > data['HSC_r_FLUX'+aper    ])
                      & (FLUXCUT > data['HSC_i_FLUX'+aper    ])
                      & (FLUXCUT > data['HSC_z_FLUX'+aper    ])
                      & (FLUXCUT > data['HSC_y_FLUX'+aper    ])
                      & (FLUXCUT > data['SC_gp_FLUX'+aper    ]) 
                      & (FLUXCUT > data['SC_rp_FLUX'+aper    ])
                      & (FLUXCUT > data['SC_ip_FLUX'+aper    ])
                      & (FLUXCUT > data['SC_zp_FLUX'+aper    ])
                      & (FLUXCUT > data['UVISTA_Y_FLUX'+aper ])
                      & (zphot2['z_spec']<0) & (zphot['type'] == 0)
                      & (data['FLAG_SUPCAM'] == 0) & (data['FLAG_HSC'] == 0) 
                      #& (data['UVISTA_NB118_FLAGS'] == 0)
                      & (data['UVISTA_J_FLAGS'] == 0)
                     )[0]
    
    ranged2 = np.where( (data['UVISTA_Ks_MAGERR'+aper]        <= 10) & (data['UVISTA_Ks_MAG'+aper]         >= 0)
                      & (data['UVISTA_J_MAGERR'+aper ]        <= 10) & (data['UVISTA_J_MAG'+aper ]         >= 0)
                      & (data['UVISTA_H_MAGERR'+aper ]        <= 10) & (data['UVISTA_H_MAG'+aper ]         >= 0)
                      #& (data['UVISTA_Ks_FLUX'+aper] >= 3*data['UVISTA_Ks_FLUXERR'+aper])
                      #& (data['UVISTA_H_FLUX'+aper] >= 3*data['UVISTA_H_FLUXERR'+aper])
                      )[0]
    
    
    ranged = np.intersect1d(ranged1,ranged2)
    

    
    
    
    Fsel = ['HSC_g','HSC_r','HSC_i','HSC_z','HSC_y','SC_gp','SC_rp','SC_ip','SC_zp','UVISTA_Y',
            'SC_IB427', 'SC_IB464', 'SC_IA484', 'SC_IB505', 'SC_IA527', 'SC_IB574', 'SC_IA624', 
            'SC_IA679', 'SC_IB709', 'SC_IA738', 'SC_IA767', 'SC_IB827', 'SC_NB711', 'SC_NB816']
    
    
    
    #Cut for narrowband
    cutn = np.where( data['UVISTA_J_FLUX'+aper][ranged] >= 3*data['UVISTA_J_FLUXERR'+aper][ranged])[0]
    
    # for i in range(len(Fsel)):
    #     FLUX    = data['%s_FLUX%s'   %(Fsel[i],aper)][ranged]
    #     FLUXERR = data['%s_FLUXERR%s'%(Fsel[i],aper)][ranged]
        
    #     cutc = np.where( (FLUX - 3*FLUXERR <= 0))
    #     cutn = np.intersect1d(cutn,cutc)
        
    FLUX = np.zeros([len(Fsel),len(ranged)])
    FLUXERR = np.zeros([len(Fsel),len(ranged)])
    for i in range(len(Fsel)):
        FLUX[i]    = data['%s_FLUX%s'   %(Fsel[i],aper)][ranged]
        FLUXERR[i] = data['%s_FLUXERR%s'%(Fsel[i],aper)][ranged]
    m,sm = wmean(FLUX,FLUXERR,axis=0)
    cutn = np.intersect1d(cutn,np.where(m - 3*sm <= 0)[0])

        
if SEDSetting == 2:
    ### Selecting hoe much larger Y flux has to be than lower wavelength fluxes (Z, V, B, J and Y)
    order = 8
    
    FLUXCUT = data['UVISTA_Y_FLUX'+aper]/order
    
    ranged1 = np.where( (FLUXCUT > data['HSC_g_FLUX'+aper    ]) 
                      & (FLUXCUT > data['HSC_r_FLUX'+aper    ])
                      & (FLUXCUT > data['HSC_i_FLUX'+aper    ])
                      & (FLUXCUT > data['HSC_z_FLUX'+aper    ])
                      & (FLUXCUT > data['HSC_y_FLUX'+aper    ])
                      & (FLUXCUT > data['SC_gp_FLUX'+aper    ]) 
                      & (FLUXCUT > data['SC_rp_FLUX'+aper    ])
                      & (FLUXCUT > data['SC_ip_FLUX'+aper    ])
                      & (FLUXCUT > data['SC_zp_FLUX'+aper    ])
                      #& (FLUXCUT > data['UVISTA_Y_FLUX'+aper ])
                      & (zphot2['z_spec']<0) & (zphot['type'] == 0)
                      & (data['FLAG_SUPCAM'] == 0) & (data['FLAG_HSC'] == 0) 
                      #& (data['UVISTA_NB118_FLAGS'] == 0)
                      & (data['UVISTA_Y_FLAGS'] == 0)
                     )[0]
    
    ranged2 = np.where( (data['UVISTA_Ks_MAGERR'+aper]        <= 10) & (data['UVISTA_Ks_MAG'+aper]         >= 0)
                      & (data['UVISTA_J_MAGERR'+aper ]        <= 10) & (data['UVISTA_J_MAG'+aper ]         >= 0)
                      & (data['UVISTA_H_MAGERR'+aper ]        <= 10) & (data['UVISTA_H_MAG'+aper ]         >= 0)
                      #& (data['UVISTA_Ks_FLUX'+aper] >= 3*data['UVISTA_Ks_FLUXERR'+aper])
                      #& (data['UVISTA_H_FLUX'+aper] >= 3*data['UVISTA_H_FLUXERR'+aper])
                      )[0]
    
    
    ranged = np.intersect1d(ranged1,ranged2)
    

    
    
    
    Fsel = ['HSC_g','HSC_r','HSC_i','HSC_z','HSC_y','SC_gp','SC_rp','SC_ip','SC_zp',
            'SC_IB427', 'SC_IB464', 'SC_IA484', 'SC_IB505', 'SC_IA527', 'SC_IB574', 'SC_IA624', 
            'SC_IA679', 'SC_IB709', 'SC_IA738', 'SC_IA767', 'SC_IB827', 'SC_NB711', 'SC_NB816']
    
    
    
    #Cut for narrowband
    cutn = np.where( data['UVISTA_Y_FLUX'+aper][ranged] >= 5*data['UVISTA_Y_FLUXERR'+aper][ranged])[0]
    
    # for i in range(len(Fsel)):
    #     FLUX    = data['%s_FLUX%s'   %(Fsel[i],aper)][ranged]
    #     FLUXERR = data['%s_FLUXERR%s'%(Fsel[i],aper)][ranged]
        
    #     cutc = np.where( (FLUX - 3*FLUXERR <= 0))
    #     cutn = np.intersect1d(cutn,cutc)
        
    FLUX = np.zeros([len(Fsel),len(ranged)])
    FLUXERR = np.zeros([len(Fsel),len(ranged)])
    for i in range(len(Fsel)):
        FLUX[i]    = data['%s_FLUX%s'   %(Fsel[i],aper)][ranged]
        FLUXERR[i] = data['%s_FLUXERR%s'%(Fsel[i],aper)][ranged]
    m,sm = wmean(FLUX,FLUXERR,axis=0)
    cutn = np.intersect1d(cutn,np.where(m - 2*sm <= 0)[0])


cutn = ranged[cutn]
if 234919 in data['ID'][cutn]:
    cutn = np.delete(cutn,np.where(234919 == data['ID'][cutn])[0][0])
#cutn = np.append(np.where(data['ID'] == 243863)[0][0],cutn)
#cutn = np.concatenate((cutn,[1389194])) #[1389194, 1216956]


Ra = data['ALPHA_J2000'][cutn]
Dec = data['DELTA_J2000'][cutn]
Size = xysize(cutn)
ID = data['ID'][cutn]

# plt.figure(figsize=(8,6),dpi=100)
# plt.hist2d(Ra,Dec,200)
# plt.gca().invert_xaxis()
# plt.colorbar()
# plt.xlabel('Ra')
# plt.ylabel('Dec')
# plt.tight_layout()




#Stefanon = ['UVISTA-Y10','no','UVISTA-Y12','no']
#Bowler = ['UVISTA–598','no','no','no']


cutno = np.array([ 784809,  895879, 1151106, 1371151])


S1 = np.where( (data['UVISTA_J_FLUX_APER2'] >= 3*data['UVISTA_J_FLUXERR_APER2']) 
              & (data['UVISTA_NB118_FLUX_APER2'] >= 5*data['UVISTA_NB118_FLUXERR_APER2'])
              & (data['UVISTA_J_FLUX_APER2']>0)
              & (data['UVISTA_NB118_MAG_APER2']>16)
              & (corr(np.arange(len(data)))>-2)
              )[0]
NB = data['UVISTA_NB118_MAG_APER2'][S1]
JNB,JNBr = corr(S1,True)
S2 = np.where( (JNB>0.2) & (JNB>=2.5*JNBr) )[0]
fig1 = plt.figure(figsize=(7,3),dpi=350)
plt.plot(NB,JNB,'ok',ms=.5,label=r'3$\sigma$ in $J$ , 5$\sigma$ in $N\!B118$')
plt.plot(NB[S2],JNB[S2],'or',ms=.5,label=r'$(J-{N\!B118})_{corr} \geq 0.2$'+'\n' 
         + r'$(J-{N\!B118})_{corr} \geq 2.5\sigma_{(J-{N\!B118})_{corr}}$')
plt.xlim([np.min(NB),np.max(NB)])
plt.ylim(bottom=-2)
plt.legend(loc='upper left')
plt.xlabel(r'$N\!B118$ mag')
plt.ylabel(r'$(J-{N\!B118})_{corr}$ mag')
fig1.subplots_adjust(top=0.998,bottom=0.173,left=0.062,right=0.995,hspace=0.2,wspace=0.2)


#%%     Select filters for spectrum and find photometric redshift in zphot if avalable and Plotting spectra and thumbnails









Fplot = ['HSC_g','HSC_r','HSC_i','HSC_z','HSC_y','UVISTA_Y','UVISTA_NB118','UVISTA_J','UVISTA_H','UVISTA_Ks',
         'SC_NB711','SC_NB816','SC_B','SC_V']#,'IRAC_CH1','IRAC_CH2']
Filters = pd.read_csv('Filters.csv',sep = ', ',engine='python')
Fav = np.zeros(len(Fplot))
Fsp = np.zeros(len(Fplot))
Fwav =np.zeros(len(Fplot))
Fwavs=np.zeros(len(Fplot))


plt.close(fig=2); plt.figure(2,figsize=(8,6),dpi=100)
for i in range(len(Fav)):
    APER = aper
    if 'IRAC' in Fplot[i] or 'GALEX' in Fplot[i] or 'FIR' in Fplot[i]:
        APER = ''
    #print('%s_FLUX%s'         % (Fplot[i],APER))
    FLUX = data['%s_FLUX%s'         % (Fplot[i],APER)][cutn]
    FLUXERR = data['%s_FLUXERR%s'   % (Fplot[i],APER)][cutn]
    Fav[i],Fsp[i] = wmean(FLUX,FLUXERR)
    s = -1 + len(Fplot[i][Fplot[i].index('_'):])
    if 'IRAC' in Fplot[i] or 'FIR' in Fplot[i]:
        s = 0

    Fwav[i]  = Filters['Mean'][np.where(Fplot[i][-s:] == Filters['Filter'])[0][0]]
    Fwavs[i] = Filters['FWHM'][np.where(Fplot[i][-s:] == Filters['Filter'])[0][0]]/2
#plt.gca().set_ylim(bottom=0)

def SigmaArrows(x,y,xr,yr,sigma=2):
    x = np.asarray(x)
    xr = np.asarray(xr)
    y = np.asarray(y)
    yr = np.asarray(yr)
    S1 = np.where( (y-sigma*yr) <= 0)[0]
    S2 = np.delete(np.arange(len(y)),S1)
    plt.errorbar(x[S2],y[S2],xerr=xr[S2],yerr=yr[S2],fmt='.g',ms=2,capsize=2)
    plt.errorbar(x[S1],y[S1]+sigma*yr[S1],0,xerr=xr[S1],fmt='.k',ms=2,capsize=2)
    for i in S1:
        plt.annotate('',xy=(x[i],y[i]),xytext=(x[i],y[i]+sigma*yr[i]),arrowprops=dict(arrowstyle="->"))
    return




# ArrSel1 = np.where(Fav-2*Fsp <= 0)[0]
# ArrSel2 = np.where(Fav-2*Fsp >  0)[0]
# plt.errorbar(Fwav[ArrSel2],Fav[ArrSel2],yerr=Fsp[ArrSel2],xerr=Fwavs[ArrSel2],fmt='.k',ms=2,capsize=2)
# for i in ArrSel1:
#     plt.annotate('',xy=(Fwav[i],Fav[i]+2*Fsp[i]),xytext=(0,-Fsp[i]),arrowprops=dict(arrowstyle="->"))
#     #plt.arrow(Fwav[i],Fav[i]+2*Fsp[i],0,-Fsp[i],head_width=200, head_length=0.015,fc='k')
#     plt.errorbar(Fwav[i],Fav[i]+2*Fsp[i], xerr=Fwavs[i],fmt='k',ms=2,capsize=2)
SigmaArrows(Fwav,Fav,Fwavs,Fsp)
Nbi = [i for i in range(len(Fplot)) if 'NB118' in Fplot[i]][0]
#plt.errorbar(Fwav[Nbi],Fav[Nbi],yerr=Fsp[Nbi],xerr=Fwavs[Nbi],fmt='.r',ms=2,capsize=2)
plt.xlabel('Wavelength [Å]')
plt.ylabel('Flux [mJy]')
plt.tight_layout()
#plt.gca().set_ylim(bottom=0)






common, i1,i2 = np.intersect1d(data['ID'][cutn],zphot['Id'],return_indices=True)
i1 = i1[[2,3,4,5,0,1,6]]
i2 = i2[[2,3,4,5,0,1,6]]
cutn = cutn[i1]
Ra = Ra[i1]
Dec = Dec[i1]
Size = Size[i1]
zcutn = np.array(zphot['zBEST'][i2])


#
plt.close(fig=3)

def subidex(n): #Find size of subplots closest to square to have n subplots
    a = np.ceil(np.sqrt(n)).astype(int)
    b = np.ceil(n/a).astype(int)
    return b,a



iwav = []
for i in range(len(Fplot)):
    s = -1 + len(Fplot[i][Fplot[i].index('_'):])
    if 'IRAC' in Fplot[i] or 'FIR' in Fplot[i]:
        s = 0
    iwav.append(np.where(Fplot[i][-s:] == Filters['Filter'])[0][0])
wave = np.array(Filters['Mean'])[iwav]
waver= np.array(Filters['FWHM'])[iwav]/2

plt.figure(3,figsize=(18,12),dpi=100)
subp = 1
for cn in cutn:
    FLUX = []
    FLUXERR = []
    for i in range(len(Fplot)):
        APER = aper
        if 'IRAC' in Fplot[i] or 'GALEX' in Fplot[i] or 'FIR' in Fplot[i]:
            APER = ''
        FLUX.append(data['%s_FLUX%s'   %(Fplot[i],APER)][cn])
        FLUXERR.append(data['%s_FLUXERR%s'   %(Fplot[i],APER)][cn])
    ax = plt.subplot(*subidex(len(cutn)),subp)
    #ax.set_yscale('log')
    
    subp += 1

    # plt.errorbar(wave,FLUX,FLUXERR,waver,fmt='.k',ms=2,capsize=2,label='source i%i'% (subp-2))
    # if zcutn[subp-2] < 0:
    #     plt.errorbar(wave[Nbi],FLUX[Nbi],FLUXERR[Nbi],waver[Nbi],fmt='.r',ms=2,capsize=2,label='N/A')
    # else:
    #     plt.errorbar(wave[Nbi],FLUX[Nbi],FLUXERR[Nbi],waver[Nbi],fmt='.r',ms=2,capsize=2,label='z=%.3g'% (zcutn[subp-2]))
    SigmaArrows(wave, FLUX, waver, FLUXERR)
    #plt.ylim([0, 1.2*np.max(FLUX)])
    #plt.legend()
    plt.xlabel('Wavelength [Å]')
    plt.ylabel('Flux [mJy]')
plt.tight_layout()




SIZE = 75

plt.close(fig=4);plt.figure(4,figsize=(12,8),dpi=100)
for i in range(len(cutn)):
    axes = plt.subplot(*subidex(len(cutn)),1+i,projection=Projection(Ra[i],Dec[i],Size[i],SIZE)) 
    ThumbNail2(Ra[i],Dec[i],Size[i],SIZE,subplot=True,axes=axes,filters=['Ks','NB118','Y'],sigma=0,stretch=5,Q=0.1)
    plt.xlabel(' ',)
    plt.ylabel(' ')
    plt.title('%i'%ID[i])
    #plt.axis('off')
plt.tight_layout()

#%%               Narrowband excess?
# if 1389194 not in cutn:
#     cutn = np.append(cutn,1389194)

#Data(cutn)
JNB, JNBr = corr(cutn,True)
NB118 = data['UVISTA_NB118_MAG'+aper ][cutn]
NBr = data['UVISTA_NB118_MAGERR'+aper ][cutn]


plt.close(fig=5)
plt.figure(5,figsize=(8,5),dpi=100)
plt.errorbar(NB118[NB118>20],JNB[NB118>20],JNBr[NB118>20],NBr[NB118>20],ms=2,capsize=2,lw=1,fmt='.k')
plt.xlabel('$NB118$ mag')
plt.ylabel('$(J-NB118)_{corr}$ mag')
#plt.xlim([24,30])
#plt.ylim([-.5,2.5])
plt.tight_layout()

# import scipy
# from scipy.optimize import curve_fit

# plt.figure()
# def CDF(x,mu,sigma):
#     return 1/2 * (1 + scipy.special.erf((x-mu) / (sigma*np.sqrt(2)) ) )
# def Gaussian(x,mu,sigma):
#     return np.exp(- (x-mu)**2 / (2*sigma**2) )/(sigma*np.sqrt(2*np.pi))
# ab,cov = curve_fit(CDF,zphot2['cdf'][cutn][3],CDF(np.linspace(-5,5,51),0,1),p0=(8,0.1))
# #plt.plot(zphot2['cdf'][cutn][3],CDF(np.linspace(-5,5,51),0,1))
# plt.plot(zphot2['cdf'][cutn][3],Gaussian(zphot2['cdf'][cutn][3],*ab))




#%%  Making better specs

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

# sel = np.array([5,9,15,17,18,20,26,30,31,33,34,35,36])
#sel = np.array([7,9,10,12])
sel = np.arange(len(cutn))
cutns = cutn[sel]
Ra2 = Ra[sel]
Dec2 = Dec[sel]
Size2 = Size[sel]

#find filters
filters = []
for i in range(len(hdr)):
    if '_MAG' in "%s" % hdr[i] and 'AUTO' not in  "%s" % hdr[i] and 'APER3' not in  "%s" % hdr[i] and 'ERR' not in "%s" % hdr[i] and 'ISO' not in "%s" % hdr[i]:
        filters.append(hdr[i])
def sort(arrs,direction = 'asc'):
    arrs = [np.asarray(x) for x in arrs]
    inds = np.argsort(arrs[0])
    if direction != 'asc':
        inds = inds[::-1]
    return [x[inds] for x in arrs]
def Magnitude(index):
    files = os.listdir('.\\Filters')

    hdf = [];hdfr= [];idxi = []
    for i in range(len(filters)):
        for y in range(len(files)):
            if filters[i][:filters[i].find('_MAG')] == files[y][:-4]:
                idxi.append(y)
                hdf.append(filters[i])
                hdfr.append(filters[i][:filters[i].find('_MAG')+4]+'ERR'+filters[i][filters[i].find('_MAG')+4:])
    wavelength = [];width = []
    for i in idxi:
        fdata = np.loadtxt('.\\Filters\%s' % files[i])
        wavelength.append( Peak(fdata[:,0],fdata[:,1]) )
        width.append( 0.5*FWHM(fdata[:,0],fdata[:,1]) )
    
    MAG = []
    MAGERR = []
    w = []
    wr = []
    hdf2 = []
    for i in range(len(idxi)):
        mag = data[hdf[i]][index]
        magerr = data[hdfr[i]][index]
        if magerr < 10 and mag >= 0:
            MAG.append(mag)
            MAGERR.append(magerr)
            w.append(wavelength[i])
            wr.append(width[i])
            hdf2.append(hdf[i])
    return sort([MAG,MAGERR,w,wr,hdf2],direction='desc')


filters2 = []
for i in range(len(hdr)):
    if '_FLUX' in "%s" % hdr[i] and 'AUTO' not in  "%s" % hdr[i] and 'APER3' not in  "%s" % hdr[i] and 'ERR' not in "%s" % hdr[i] and 'ISO' not in "%s" % hdr[i]:
        filters2.append(hdr[i])
def Fluxxer(index):
    files = os.listdir('.\\Filters')

    hdf = [];hdfr= [];idxi = []
    for i in range(len(filters2)):
        for y in range(len(files)):
            if filters2[i][:filters2[i].find('_FLUX')] == files[y][:-4]:
                idxi.append(y)
                hdf.append(filters2[i])
                hdfr.append(filters2[i][:filters2[i].find('_FLUX')+5]+'ERR'+filters2[i][filters2[i].find('_FLUX')+5:])
    wavelength = [];width = []
    for i in idxi:
        fdata = np.loadtxt('.\\Filters\%s' % files[i])
        wavelength.append( Peak(fdata[:,0],fdata[:,1]) )
        width.append( 0.5*FWHM(fdata[:,0],fdata[:,1]) )
    
    FLUX = []
    FLUXERR = []
    w = []
    wr = []
    hdf2 = []
    for i in range(len(idxi)):
        flux = data[hdf[i]][index]
        fluxerr = data[hdfr[i]][index]
        if fluxerr < 1 and flux < 50 and flux > -90 and flux>0:
            FLUX.append(flux)
            FLUXERR.append(fluxerr)
            w.append(wavelength[i])
            wr.append(width[i])
            hdf2.append(hdf[i])
    return sort([FLUX,FLUXERR,w,wr,hdf2])


filters2F = []
for i in range(len(hdrF)):
    if '_FLUX' in "%s" % hdrF[i] and 'AUTO' not in  "%s" % hdrF[i] and 'APER3' not in  "%s" % hdrF[i] and 'ERR' not in "%s" % hdrF[i] and 'ISO' not in "%s" % hdrF[i]:
        filters2F.append(hdrF[i])
def FluxxerF(index):
    files = os.listdir('.\\Filters')

    hdf = [];hdfr= [];idxi = []
    for i in range(len(filters2F)):
        for y in range(len(files)):
            if filters2F[i][:filters2F[i].find('_FLUX')] == files[y][:-4]:
                idxi.append(y)
                hdf.append(filters2F[i])
                hdfr.append(filters2F[i][:filters2F[i].find('_FLUX')+5]+'ERR'+filters2F[i][filters2F[i].find('_FLUX')+5:])
    wavelength = [];width = []
    for i in idxi:
        fdata = np.loadtxt('.\\Filters\%s' % files[i])
        wavelength.append( Peak(fdata[:,0],fdata[:,1]) )
        width.append( 0.5*FWHM(fdata[:,0],fdata[:,1]) )
    
    FLUX = []
    FLUXERR = []
    w = []
    wr = []
    hdf2 = []
    for i in range(len(idxi)):
        flux = dataF[hdf[i]][index]
        fluxerr = dataF[hdfr[i]][index]
        if fluxerr < 1 and flux < 50 and flux > -90:
            FLUX.append(flux)
            FLUXERR.append(fluxerr)
            w.append(wavelength[i])
            wr.append(width[i])
            hdf2.append(hdf[i])
    return sort([FLUX,FLUXERR,w,wr,hdf2])


def SigmaArrowsMAG(index,sigma=2):
    def Magnitude2(index):
        files = os.listdir('.\\Filters')
    
        hdf = [];hdfr= [];idxi = []
        for i in range(len(filters)):
            for y in range(len(files)):
                if filters[i][:filters[i].find('_MAG')] == files[y][:-4]:
                    idxi.append(y)
                    hdf.append(filters[i])
                    hdfr.append(filters[i][:filters[i].find('_MAG')+4]+'ERR'+filters[i][filters[i].find('_MAG')+4:])
        wavelength = [];width = []
        for i in idxi:
            fdata = np.loadtxt('.\\Filters\%s' % files[i])
            wavelength.append( Peak(fdata[:,0],fdata[:,1]) )
            width.append( 0.5*FWHM(fdata[:,0],fdata[:,1]) )
        
        MAG = []
        MAGERR = []
        w = []
        wr = []
        hdf2 = []
        FLUX = []
        FLUXERR = []
        def fluxfinder(magstring):
            index = magstring.find('_MAG')
            s1 = magstring[:index]
            s2 = magstring[index+4:]
            return s1+'_FLUX'+s2
        
        for i in range(len(idxi)):
            mag = data[hdf[i]][index]
            magerr = data[hdfr[i]][index]
            flux = data[fluxfinder(hdf[i])][index]
            fluxerr = data[fluxfinder(hdfr[i])][index]
            if mag > 0 and mag < 40 and magerr < 5:
                MAG.append(mag)
                MAGERR.append(magerr)
                w.append(wavelength[i])
                wr.append(width[i])
                hdf2.append(hdf[i])
                FLUX.append(flux)
                FLUXERR.append(fluxerr)
        return sort([MAG,MAGERR,w,wr,hdf2,FLUX,FLUXERR],direction='desc')

    MAG,MAGERR,x,xr,hdf2,y,yr = Magnitude2(index)
    x = np.asarray(x)
    xr = np.asarray(xr)
    y = np.asarray(y)
    yr = np.asarray(yr)
    S1 = np.where( (y-sigma*yr) <= 0)[0]
    S2 = np.delete(np.arange(len(y)),S1)
    if len(S2)>0:
        plt.errorbar(x[S2],MAG[S2],xerr=xr[S2],yerr=MAGERR[S2],fmt='.k',ms=2,capsize=2)
    plt.errorbar(x[S1],MAG[S1]-sigma*MAGERR[S1],0,xerr=xr[S1],fmt='.k',ms=2,capsize=2)
    
    LowerLim = MAG-sigma*MAGERR
    LowerLim[LowerLim > 30] = 30
    for i in S1:
        plt.annotate('',xy=(x[i],MAG[i]),xytext=(x[i],LowerLim[i]),arrowprops=dict(arrowstyle="->"))
    return






def SigmaArrows(x,y,xr,yr,sigma=2,arrow="<-",ymin=1e-1):
    Sp = np.where(y>=0)
    x = np.asarray(x)[Sp]
    xr = np.asarray(xr)[Sp]
    y = np.asarray(y)[Sp]
    yr = np.asarray(yr)[Sp]
    S1 = np.where( (y-sigma*yr) <= 0)[0]
    S2 = np.delete(np.arange(len(y)),S1)
    plt.errorbar(x[S2],y[S2],xerr=xr[S2],yerr=yr[S2],fmt='.g',ms=2,capsize=2)
    plt.errorbar(x[S1],y[S1]+sigma*yr[S1],yerr=0,xerr=xr[S1],fmt='.k',ms=2,capsize=2)
    ye = y[S1]
    for p,i in enumerate(S1):
        if ye[p] < ymin:
            ye[p] = ymin
        plt.annotate('',xy=(x[i],ye[p]+sigma*yr[i]),xytext=(x[i],ye[p]),arrowprops=dict(arrowstyle=arrow))
    try:  
        plt.ylim(bottom=np.min(ye))
    except:
        print('Failed to set mag limit')

    return






plt.close(fig=6)
plt.figure(6,figsize=(24,12),dpi=100)
subp = 1
for index in cutns:
    ax = plt.subplot(*subidex(len(cutns)),subp)
    plt.gca().invert_yaxis()
    ax.set_xscale('log')
    #ax.set_yscale('log')
    subp += 1

    # MAG,MAGERR,w,wr,hdf2 = Magnitude(index)
    # plt.errorbar(w,MAG,MAGERR,wr,fmt='.k',lw=1,ms=2,capsize=2,label='Source%i' %  (subp-2))
    # inbval = False
    # for i in range(len(hdf2)):
    #     if 'NB118' in hdf2[i]:
    #         inbval = True
    # if inbval:
    #     inb = [i for i in range(len(hdf2)) if 'NB118' in hdf2[i]][0]
    #     if zcutn[subp-2] < 0:
    #         plt.errorbar(w[inb],MAG[inb],MAGERR[inb],wr[inb],fmt='.r',lw=1,ms=2,capsize=2,label='N/A')
    #     else:
    #         plt.errorbar(w[inb],MAG[inb],MAGERR[inb],wr[inb],fmt='.r',lw=1,ms=2,capsize=2,label='z=%.3g' % (zcutn[subp-2]))
    SigmaArrowsMAG(index,2)

    
    #plt.xlabel('Wavelength [Å]')
    #plt.ylabel('Flux [mJy]')
    #plt.legend()
    #plt.xlim([1e+3,1e+5])
plt.tight_layout()





plt.close(fig=7)
plt.figure(7,figsize=(13,6),dpi=100)
subp = 1
for index in cutns:
    ax = plt.subplot(*subidex(len(cutns)),subp)
    subp+=1
    
    FLUX,FLUXERR,w,wr,hdf2 = Fluxxer(index)
    SigmaArrows(w/10000,FLUX,wr/10000,FLUXERR,2.,ymin=5e-2)
    ax.set_xscale('log')
    plt.xlabel(r'Wavelength [$\mu$m]')
    plt.ylabel(r'Flux [$\mu$Jy]')
    ticks = [0.3,0.5,1,2,5]
    #ticks = ticks[np.where(ticks < min(w/10000))[0][-1]:]
    #ticks = np.asarray(ticks[:np.where(ticks > max(w/10000))[0][0]+1])
    labels = ['%.2g'%i for i in ticks]
    plt.xticks(ticks,labels)
    plt.gca().set_yscale('log')
plt.tight_layout()

#%%

Size = xysize(cutn)
Ra = data['ALPHA_J2000'][cutn]
Dec = data['DELTA_J2000'][cutn]
ID = data['ID'][cutn]
i_FARMER = LocateFarmer(Ra, Dec)
RaF = dataF['ALPHA_J2000'][i_FARMER]
DecF = dataF['DELTA_J2000'][i_FARMER]
ID_F = dataF['ID'][i_FARMER]



def cdf(cdfz,Np=1001):
    def GaussPDF(z=None,mu=0,sigma=1):
        x = np.linspace(-5,5,51)
        a = np.sqrt(2*np.pi)*sigma
        return np.exp(-(x-mu)**2 / (2*sigma**2) )/a
    z_new = np.linspace(cdfz[0],cdfz[-1],Np)
    for i in range(len(cdfz)-1):
        if cdfz[i] == cdfz[i+1]:
            cdfz[i+1:] += 1e-8
    f = interp1d(cdfz,GaussPDF(),kind='cubic')
    y = f(z_new)/(np.trapz(f(z_new),z_new))
    return z_new,np.abs(y)
pdzCL = {} #Classic Lephare
pdzCE = {} #Classic Eazy
pdzFL = {} #Farmer Lephare
pdzFE = {} #Farmer Eazy
z_LePhare = np.linspace(0,10,1001)
z_EAZY = fits.open('farmer_v051120.full.data.fits')[2].data

# from scipy.stats import chi2
# def PDZ(chi,z=z_EAZY):
#     pdz = chi2(chi).pdf(z)
#     norm = np.trapz(pdz,z)
#     return pdz/norm



hdul = fits.open('photoz_cosmos2020_lephare_classic_v1.8.pdz.fits')
for i in range(len(cutn)):
    pdzCL['%i'%cutn[i]] = hdul[0].data[cutn[i]]
hdul.close()

hdul = fits.open('photoz_cosmos2020_lephare_farmer_v1.8.pdz.fits')
for i in range(len(cutn)):
    pdzFL['%i'%i_FARMER[i]] = hdul[0].data[i_FARMER[i]]
hdul.close()

hdul = fits.open('classic_v1.5.full.data.fits')
for i in range(len(cutn)):
    pdzCE['%i'%cutn[i]] = cdf(zphot2['cdf'][cutn[i]])
hdul.close()

hdul = fits.open('farmer_v051120.full.data.fits')
for i in range(len(cutn)):
    pdzFE['%i'%i_FARMER[i]] = cdf(zphot2F['cdf'][i_FARMER[i]])
hdul.close()

# hdul = fits.open('classic_v1.5.full.data.fits')
# for i in range(len(cutn)):
#     pdzCE['%i'%cutn[i]] = PDZ(hdul[3].data[cutn[i]])
# hdul.close()

# hdul = fits.open('farmer_v051120.full.data.fits')
# for i in range(len(cutn)):
#     pdzFE['%i'%i_FARMER[i]] = PDZ(hdul[3].data[i_FARMER[i]])
# hdul.close()





def FluxToMag(F,Fr,AB0=21.39827018,a=0.09984529):
    M = AB0 - 2.5*np.log10(F*a)
    AB0r = 0.0102827
    ar = 0.00094285
    a1 = 0.4342944819
    a2 = 5.301898109
    A1 = (6.26*Fr**2+a2*F**2*AB0r**2)*a**2
    A2 = 6.25*ar**2*F**2
    A3 = F**2*a**2
    return M,a1*np.sqrt( (A1+A2) / A3 )
#%%
def zFits(Classic_Index,zformat='%.4g',zrformat='%.4g'):
    cutC = Classic_Index
    Ra = data['ALPHA_J2000'][cutC]
    Dec= data['DELTA_J2000'][cutC]
    cutF  =  LocateFarmer(Ra, Dec)
    
    LPC  = zphot['zPDF']     [cutC]
    LPCl = zphot['zPDF_l68'] [cutC]
    LPCu = zphot['zPDF_u68'] [cutC]
    LPF  = zphotF['zPDF']    [cutF]
    LPFl = zphotF['zPDF_l68'][cutF]
    LPFu = zphotF['zPDF_u68'][cutF]
    
    EZC =  zphot2['z500'] [cutC]
    EZCl = zphot2['z160'] [cutC]
    EZCu = zphot2['z840'] [cutC]
    EZF =  zphot2F['z500'][cutF]
    EZFl = zphot2F['z160'][cutF]
    EZFu = zphot2F['z840'][cutF]
    
    
    CLP = r'$' + zformat % LPC + '_{' + zrformat % (LPC-LPCl) + '}' + '^{' + zrformat % (LPCu-LPC) + '}' + '$'
    FLP = r'$' + zformat % LPF + '_{' + zrformat % (LPF-LPFl) + '}' + '^{' + zrformat % (LPFu-LPF) + '}' + '$'
    CEZ = r'$' + zformat % EZC + '_{' + zrformat % (EZC-EZCl) + '}' + '^{' + zrformat % (EZCu-EZC) + '}' + '$'
    FEZ = r'$' + zformat % EZF + '_{' + zrformat % (EZF-EZFl) + '}' + '^{' + zrformat % (EZFu-EZF) + '}' + '$'
    return CLP,FLP,CEZ,FEZ
    
# cutn = np.delete(cutn,-2)


# ar1 = []
# ar2 = []
# ar3 = []
# ar4 = []
# for i in range(len(cutn)):
#     l1,l2,e1,e2 = zFits(cutn[i],'%.3g','%.1g')
#     ar1.append(l1)
#     ar2.append(e1)
#     ar3.append(l2)
#     ar4.append(e2)

def FluxToMagSigmaArrows(x,y,xr,yr,hdf2,cut,sigma=2,arrow="<-",alpha=0.6,annotation=False):
    x = np.asarray(x)
    xr = np.asarray(xr)
    y = np.asarray(y)
    yr = np.asarray(yr)
    mag = np.zeros(len(hdf2))
    magerr = np.zeros(len(hdf2))
    #mag,magerr = FluxToMag(y, yr)
    for i in range(len(hdf2)):
        try:
            magc = hdf2[i].replace('_FLUX','_MAG')
            magcr=hdf2[i].replace('_FLUX','_MAGERR')
            mag[i] = data[magc][cut]
            magerr[i] = data[magcr][cut]
        except:
            mag[i],magerr[i] = FluxToMag(y[i],yr[i])
    S1 = np.where( ((y-sigma*yr) <= 0) )[0]
    S2 = np.delete(np.arange(len(y)),S1)
    S1 = S1[np.where((mag[S1] > 15) & (mag[S1] < 40) & (magerr[S1]<5) )[0]]
    S2 = S2[np.where((mag[S2] > 15) & (mag[S2] < 40) & (magerr[S2]<5) )[0]]
    
    plt.errorbar(x[S2],mag[S2],xerr=xr[S2],yerr=magerr[S2],fmt='.g',ms=2,capsize=2)
    plt.errorbar(x[S1],mag[S1]-sigma*magerr[S1],yerr=0,xerr=xr[S1],fmt='.k',ms=2,capsize=2,alpha=alpha)
    for i in S1:
        plt.annotate( '' ,xytext=(x[i],mag[i]),xy=(x[i],mag[i]-sigma*magerr[i]),arrowprops=dict(arrowstyle=arrow,alpha=alpha))
        if annotation:
            plt.annotate( '%s'%(hdf2[i][:hdf2[i].find('_FLUX')]) ,xy=(x[i],mag[i]-sigma*magerr[i]*0),alpha=alpha, ha='center')
    for i in S2: 
        if annotation:
            plt.annotate( '%s'%(hdf2[i][:hdf2[i].find('_FLUX')]) ,xy=(x[i],mag[i]-magerr[i]*0),alpha=alpha, ha='center')
    try:       
        plt.ylim(top=np.max(mag[S1]))
    except:
        print('Failed to set mag limit')
    return

def FluxToMagSigmaArrowsF(x,y,xr,yr,hdf2,cut,sigma=2,arrow="<-",alpha=0.6,annotation=False):
    x = np.asarray(x)
    xr = np.asarray(xr)
    y = np.asarray(y)
    yr = np.asarray(yr)
    mag = np.zeros(len(hdf2))
    magerr = np.zeros(len(hdf2))
    #mag,magerr = FluxToMag(y, yr)
    for i in range(len(hdf2)):
        try:
            magc = hdf2[i].replace('_FLUX','_MAG')
            magcr=hdf2[i].replace('_FLUX','_MAGERR')
            mag[i] = dataF[magc][cut]
            magerr[i] = dataF[magcr][cut]
        except:
            mag[i],magerr[i] = FluxToMag(y[i],yr[i])
    S1 = np.where( ((y-sigma*yr) <= 0) )[0]
    S2 = np.delete(np.arange(len(y)),S1)
    S1 = S1[np.where((mag[S1] > 15) & (mag[S1] < 40) & (magerr[S1]<5) )[0]]
    S2 = S2[np.where((mag[S2] > 15) & (mag[S2] < 40) & (magerr[S2]<5) )[0]]
    
    plt.errorbar(x[S2],mag[S2],xerr=xr[S2],yerr=magerr[S2],fmt='.g',ms=2,capsize=2)
    plt.errorbar(x[S1],mag[S1]-sigma*magerr[S1],yerr=0,xerr=xr[S1],fmt='.k',ms=2,capsize=2,alpha=alpha)
    for i in S1:
        plt.annotate( '' ,xytext=(x[i],mag[i]),xy=(x[i],mag[i]-sigma*magerr[i]),arrowprops=dict(arrowstyle=arrow,alpha=alpha))
        if annotation:
            plt.annotate( '%s'%(hdf2[i][:hdf2[i].find('_FLUX')]) ,xy=(x[i],mag[i]-sigma*magerr[i]*0),alpha=alpha, ha='center')
    for i in S2: 
        if annotation:
            plt.annotate( '%s'%(hdf2[i][:hdf2[i].find('_FLUX')]) ,xy=(x[i],mag[i]-magerr[i]*0),alpha=alpha, ha='center')
    plt.ylim(top=np.max(mag))
    return


def pdflines(filts=['Y','J','H'],y=0.2,alpha=.4):
    for i in range(len(filts)):
        i_f = np.where(Filters['Filter'] == filts[i])[0][0]
        w = Filters['Mean'][i_f]
        sig = Filters['FWHM'][i_f]/2
        z,zr = zTarget(1215.67,0,w,sig)
        z1 = z-zr
        z2 = z+zr
        plt.fill_between([z1,z2], [0,0],[y,y],alpha=alpha)
        plt.annotate(filts[i], (z,y/8), ha='center')

for i in range(len(cutn)):
    plt.close(fig=8+i)
    plt.figure(8+i,figsize=(16,5),dpi=100)
    
    NV = 1.75

    index = cutn[i]
    ax = plt.axes([0.07/NV,0.092,0.71/NV,0.85])

    
    FLUX,FLUXERR,w,wr,hdf2 = Fluxxer(index)
    #print(hdf2)
    FluxToMagSigmaArrows(w/10000,FLUX,wr/10000,FLUXERR,hdf2,index,sigma=2,arrow="<-",annotation=True)
    ax.set_xscale('log')
    plt.xlabel(r'Wavelength [$\mu$m]')
    plt.ylabel('Magnitude [mag]')
    ticks = [0.03,0.05,0.1,0.2,0.3,0.5,1,2,5]
    ticks = ticks[np.where(ticks < min(w/10000))[0][-1]:]
    #ticks = np.asarray(ticks[:np.where(ticks > max(w/10000))[0][0]+1])
    labels = ['%.10g'%i for i in ticks]
    plt.xticks(ticks,labels)
    plt.gca().invert_yaxis()
    extender = "     $(J-NB$118$)_{corr}=%.2g$ $\pm$ $%.2g$"%(JNB[i],JNBr[i])
    if JNB[i] > 20:# or JNB[i] < 0:
        plt.title(r'CLASSIC:   $z_{LePhare}=$%s   $z_{EAZY}=$%s' % (zFits(cutn[i],'%.3g','%.2g')[::2]) 
                  + '   ID: %i' % ID[i]
                  ,size=10 )
    else:
        plt.title(r'CLASSIC:   $z_{LePhare}=$%s   $z_{EAZY}=$%s' % (zFits(cutn[i],'%.3g','%.2g')[::2]) +extender 
                  + '   ID: %i' % ID[i]
                  ,size=10 )
    
    NB118LOC = [i for i in range(len(hdf2)) if 'UVISTA_NB118' in hdf2[i]]
    
    
    ax= plt.axes([0.07/NV+.55,0.092,0.71/NV,0.85])
    
    FLUX,FLUXERR,w,wr,hdf2 = FluxxerF(i_FARMER[i])
    FluxToMagSigmaArrowsF(w/10000,FLUX,wr/10000,FLUXERR,hdf2,i_FARMER[i],sigma=2,arrow="<-",annotation=True)
    ax.set_xscale('log')
    plt.xlabel(r'Wavelength [$\mu$m]')
    plt.ylabel('Magnitude [mag]')
    ticks = [0.03,0.05,0.1,0.2,0.3,0.5,1,2,5]
    ticks = ticks[np.where(ticks < min(w/10000))[0][-1]:]
    #ticks = np.asarray(ticks[:np.where(ticks > max(w/10000))[0][0]+1])
    labels = ['%.10g'%i for i in ticks]
    plt.xticks(ticks,labels)
    plt.gca().invert_yaxis()

    plt.title(r'FARMER:   $z_{LePhare}=$%s   $z_{EAZY}=$%s' % (zFits(cutn[i],'%.3g','%.2g')[1::2]) 
              + '   ID: %i' % ID_F[i]
              ,size=10 )
    
    
    

    
    SIZE = 40
    y0 = .02
    y1 = .65
    if len(NB118LOC) == 0:
        Npe = 3
    else:
        Npe = 4
        y1 = y1 + (y1-y0)/Npe-0.05
    axw = (y1-y0)/Npe-0.01
    x0 = 1-axw-0.025
    ring = 1
    #for irac stretch=0.8,Q=1
    ax = plt.axes([x0/NV,y0,axw/NV,axw])#,projection=Projection(Ra[i],Dec[i],Size[i],SIZE))
    ThumbNail2(Ra[i],Dec[i],Size[i]*ring,SIZE,subplot=True,axes=ax,filters=['Ks','J','Y'],sigma=0,stretch=2,Q=0.1)
    plt.xticks([])
    plt.yticks([])
    plt.ylabel('$Y$, $J$ and $K_s$')
    
    ax = plt.axes([x0/NV,y0+(y1-y0)/Npe*1,axw/NV,axw])#,projection=Projection(Ra[i],Dec[i],Size[i],SIZE))
    ThumbNail2(Ra[i],Dec[i],Size[i]*ring,SIZE,subplot=True,axes=ax,filters=['Ks','Ks','Ks'],sigma=0,stretch=2,Q=0.1)
    plt.xticks([])
    plt.yticks([])
    plt.ylabel('$K_s$')
    
    ax = plt.axes([x0/NV,y0+(y1-y0)/Npe*2,axw/NV,axw])#,projection=Projection(Ra[i],Dec[i],Size[i],SIZE))
    ThumbNail2(Ra[i],Dec[i],Size[i]*ring,SIZE,subplot=True,axes=ax,filters=['J','J','J'],sigma=0,stretch=2,Q=0.1)
    plt.xticks([])
    plt.yticks([])
    plt.ylabel('$J$')
    
    ax = plt.axes([x0/NV,y0+(y1-y0)/Npe*3,axw/NV,axw])#,projection=Projection(Ra[i],Dec[i],Size[i],SIZE))
    ThumbNail2(Ra[i],Dec[i],Size[i]*ring,SIZE,subplot=True,axes=ax,filters=['Y','Y','Y'],sigma=0,stretch=2,Q=0.1)
    plt.xticks([])
    plt.yticks([])
    plt.ylabel('$Y$')
    if Npe == 3:
        plt.title('6"')
    
    if Npe == 4:
        ax = plt.axes([x0/NV,y0+(y1-y0)/Npe*4,axw/NV,axw])#,projection=Projection(Ra[i],Dec[i],Size[i],SIZE))
        ThumbNail2(Ra[i],Dec[i],Size[i]*ring,SIZE,subplot=True,axes=ax,filters=['NB118','NB118','NB118'],sigma=0,stretch=2,Q=0.1)
        plt.xticks([])
        plt.yticks([])
        plt.ylabel('$NB118$')
        if Npe == 4:
            plt.title('6"')
            
            
    
    UP1 = [2,7]
    UP2 = []
    
    zlow = [6,0,7,5,0,0,0,0,0]
    ztop = 10.5
    #ztop = .35

    
    if i in UP1:
        ax = plt.axes([.575/NV,0.59,0.2/NV,0.3])
    else:
        ax = plt.axes([.575/NV,0.15,0.2/NV,0.3])
    
    z_EAZY,pdz = pdzCE['%i'%cutn[i]]
    plt.plot(z_EAZY,pdz,'b',label='EZ')
    
    pdz2 = pdzCL['%i'%cutn[i]]
    plt.plot(z_LePhare,pdz2,'r',label='LP')
    pdflines(y=np.max(np.concatenate((pdz2,pdz)))*1.2)
    ticks = [1,3,5,7,9,11]
    labels = ['%i'%i for i in ticks]
    #plt.xticks(ticks,labels)
    plt.xlim([zlow[i],ztop])
    plt.legend(loc='upper left')
    plt.title(r'pdf($z$)',size=10)
    #plt.xlabel('z')
    

    
    
    if i not in UP2:
        ax = plt.axes([.575/NV+0.55,0.15,0.2/NV,0.3],alpha=0.2)
    else:
        ax = plt.axes([.575/NV+0.55,0.60,0.2/NV,0.3],alpha=0.2)
    
    z_EAZY,pdz = pdzFE['%i'%i_FARMER[i]]
    plt.plot(z_EAZY,pdz,'b',label='EZ')
    
    pdz2 = pdzFL['%i'%i_FARMER[i]]
    plt.plot(z_LePhare,pdz2,'r',label='LP')
    pdflines(y=np.max(np.concatenate((pdz2,pdz)))*1.2)
    ticks = [1,3,5,7,9,11]
    labels = ['%i'%i for i in ticks]
    #plt.xticks(ticks,labels)
    plt.xlim([zlow[i],ztop])
    plt.legend(loc='upper left')
    plt.title(r'pdf($z$)',size=10)
    #plt.xlabel('z')
    
    
#plt.tight_layout()

def Pzphot(i,zmin):
    cC = cutn[i]
    cF = i_FARMER[i]
    sLP = np.where(z_LePhare>=zmin)[0]
    
    P_C_LP = np.trapz(pdzCL['%i'%cC][sLP],z_LePhare[sLP])
    P_F_LP = np.trapz(pdzFL['%i'%cF][sLP],z_LePhare[sLP])
    
    ze,pdz = pdzCE['%i'%cC]
    sEZ = np.where(ze >= zmin)[0]
    P_C_EZ = np.trapz(pdz[sEZ],ze[sEZ])
    
    ze,pdz = pdzFE['%i'%cF]
    sEZ = np.where(ze >= zmin)[0]
    P_F_EZ = np.trapz(pdz[sEZ],ze[sEZ])
    return P_C_LP*100 , P_C_EZ*100 , P_F_LP*100, P_F_EZ*100





#%% Plots EAZY SED means

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

root = './EAZYtemplates/'
files = []
hdree = fits.open('classic_v1.5.full.data.fits')[4].header
for i in range(17):
    files.append(root+hdree['TEMP00%.2i'%(i)])
REDSHIFT = 0.101
wave = fits.open(files[0])[1].data['wave']*(1+REDSHIFT)
raaa = [0.3,5]#[(11907-56)/10000,(11907+56)/10000]
sel = np.arange(len(wave))
sel = np.where( (wave>raaa[0]*10000) & (wave < raaa[1]*10000) )[0]
coeffs = fits.open('classic_v1.5.full.data.fits')[4].data
flux = {}
plt.figure()
for i in range(len(files)):
    plt.subplot(5,4,1+i)
    flux[ 'TEMP00%.2i' % (i) ] = np.mean(fits.open(files[i])[1].data['flux'],axis=1)
    plt.semilogx(wave[sel]/10000,-2.5*np.log10(flux[ 'TEMP00%.2i' % (i) ][sel]))
    ticks = [0.03,0.05,0.1,0.2,0.3,0.5,1,2,5,10,20,50,100,200,500,1000,2000,5000,10000]
    ticks = ticks[np.where(ticks < np.min(raaa))[0][-1]:]
    ticks = np.asarray(ticks[:np.where(ticks > np.max(raaa))[0][0]+1])
    labels = ['%.10g'%i for i in ticks]
    plt.xticks(ticks,labels)
    plt.xlim([np.min(raaa),np.max(raaa)])
    plt.gca().invert_yaxis()
    #plt.ylim(bottom=1e-11)






#%%




if SEDSetting == 0:
    folder=".//Specs/NB/"
if SEDSetting == 1:
    folder=".//Specs/J/"
if SEDSetting == 2:
    folder=".//Specs/Y/"
OverWrite = True
Indexing = True

for i in range(len(cutn)):
    if 'Source%iID%i.png' % (i,ID[i]) not in os.listdir(path=folder) or OverWrite:
        fig = plt.figure(77,figsize=(10,11),dpi=60)
        ax1 = plt.subplot(212)
        # MAG,MAGERR,w,wr,hdf2 = Magnitude2(i_FARMER[i])
        # ax1.errorbar(w,MAG,MAGERR,wr,fmt='.b',lw=1,ms=2,capsize=2,label='FARMER')
        MAG,MAGERR,w,wr,hdf2 = Magnitude(cutn[i])
        ax1.errorbar(w,MAG,MAGERR,wr,fmt='.k',lw=1,ms=2,capsize=2,label='CLASSIC')

        
        for i2 in range(len(hdf2)):
            if 'NB118' in hdf2[i2]:
                inb = [i for i in range(len(hdf2)) if 'NB118' in hdf2[i]][0]
                ax1.errorbar(w[inb],MAG[inb],MAGERR[inb],wr[inb],fmt='.g',lw=1,ms=2,capsize=2)
        
        ax1.set_xscale('log')
        plt.gca().invert_yaxis()
        plt.xlabel('Wavelength [Å]')
        plt.ylabel('Magnitude [mag]')
        
        if Size[i] <= 1:
            Size[i] = 1
        SIZE = np.int(np.ceil(8*Size[i]))
        
        axes = plt.subplot(221,projection=Projection(Ra[i],Dec[i],0,SIZE))

        #ThumbNail2(Ra[i],Dec[i],0,SIZE,addRA=[Ra[i],RaF[i]],addDEC = [Dec[i],DecF[i]],addSize=[Size[i],Size[i]],subplot=True,axes=axes,filters=['Ks','NB118','Y'],sigma=0)
        if Indexing: plt.title('Ks, NB118 and Y     ' + r'ID = %i   index=%i' % (ID[i],i))
        else:        plt.title('Ks, NB118 and Y     ' + r'ID = %i'            %  ID[i] )
        plt.xlabel('Ra')
        plt.ylabel('Dec')
        
        
        
        axes = plt.subplot(222,projection=Projection(Ra[i],Dec[i],0,SIZE))

        #ThumbNail2(Ra[i],Dec[i],0,SIZE,addRA=[Ra[i],RaF[i]],addDEC = [Dec[i],DecF[i]],addSize=[Size[i],Size[i]],subplot=True,axes=axes,filters=['Ks','J','Y'],Q=0.1,stretch=2)
        plt.title( 'Ks, J and Y     ' + RaDeStr(Ra[i],Dec[i])[0] +' , '+ RaDeStr(Ra[i],Dec[i])[1] )
        plt.xlabel('Ra')
        plt.ylabel('Dec')

        
        supplot = r'CLASSIC:   $z_{LePhare}=%.4g_{-%.2g}^{+%.2g}$  ,  $z_{EAZY}=%.4g_{-%.2g}^{+%.2g}$' % (zphot['zBEST'][cutn[i]]
                 ,zphot['zBEST'][cutn[i]]-zphot['zPDF_l68'][cutn[i]],zphot['zPDF_u68'][cutn[i]]-zphot['zBEST'][cutn[i]]
                 ,zphot2['z_min_risk'][cutn[i]],zphot2['z_min_risk'][cutn[i]]-zphot2['z160'][cutn[i]]
                 ,zphot2['z840'][cutn[i]]-zphot2['z_min_risk'][cutn[i]]
                 )
        supplot2= r'FARMER:    $z_{LePhare}=%.4g_{-%.2g}^{+%.2g}$  ,  $z_{EAZY}=%.4g_{-%.2g}^{+%.2g}$' % (zphotF['zBEST'][i_FARMER[i]]
                 ,zphotF['zBEST'][i_FARMER[i]]-zphotF['zPDF_l68'][i_FARMER[i]],zphotF['zPDF_u68'][i_FARMER[i]]-zphotF['zBEST'][i_FARMER[i]]
                 ,zphot2F['z_min_risk'][i_FARMER[i]],zphot2F['z_min_risk'][i_FARMER[i]]-zphot2F['z160'][i_FARMER[i]]
                 ,zphot2F['z840'][i_FARMER[i]]-zphot2F['z_min_risk'][i_FARMER[i]]
                 )
        
        fig.subplots_adjust(top=0.88)
        if detected_in_farmer(cutn[i], i_FARMER[i]):
            plt.suptitle(supplot+'\n'+supplot2,fontsize=20)
        else:
            plt.suptitle(supplot,fontsize=20)
        plt.tight_layout()
        plt.savefig(folder+'Source%iID%i.png' % (i,ID[i]),dpi=200)
        plt.close(fig=77)
        if len(cutn) > 10 and i % (len(cutn)//10) == 0:
            print('%.3g%% done' % (i/len(cutn)*100))





#%%


for i in range(len(hdr)):
    search = 'FLUX'
    if search in "%s" % hdr[i] and 'APER2' not in  "%s" % hdr[i] and 'APER3' not in  "%s" % hdr[i] and 'ERR' not in "%s" % hdr[i]:
        print("%s" % hdr[i])














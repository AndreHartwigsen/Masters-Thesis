#%%    Loading data and functions
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
from astropy.io import fits
import pandas as pd
from scipy.ndimage import gaussian_filter
from astropy.visualization import make_lupton_rgb




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
def zTarget(target,sigma_target=None): #All in angstrom
    nb = 11850
    snb = 20
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



def xysize(cut,cat='S'):   #Determine the radius in pixels around the central pixel that detection was performed
    # a = 40000**2/(17940*11408)
    # npix = data['FLUX_RADIUS'][cut]
    # return np.sqrt(npix/(np.pi*a))*15
    if cat == 'S':
        return np.abs(data['FLUX_RADIUS'][cut])*2
    else:
        return np.abs(dataF['FLUX_RADIUS'][cut])*2
def subidex(n): #Find size of subplots closest to square to have n subplots
    a = np.ceil(np.sqrt(n)).astype(int)
    b = np.ceil(n/a).astype(int)
    return b,a


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

aper = '_APER2'




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
        if mag <= 100 and magerr < 10 and mag>0:
            MAG.append(mag)
            MAGERR.append(magerr)
            w.append(wavelength[i])
            wr.append(width[i])
            hdf2.append(hdf[i])
    return sort([MAG,MAGERR,w,wr,hdf2])




filters2 = []
for i in range(len(hdrF)):
    if '_MAG' in "%s" % hdrF[i] and 'AUTO' not in  "%s" % hdrF[i] and 'APER3' not in  "%s" % hdrF[i] and 'ERR' not in "%s" % hdrF[i] and 'ISO' not in "%s" % hdrF[i]:
        filters2.append(hdrF[i])
def Magnitude2(index):
    files = os.listdir('.\\Filters')

    hdf = [];hdfr= [];idxi = []
    for i in range(len(filters2)):
        for y in range(len(files)):
            if filters2[i][:filters2[i].find('_MAG')] == files[y][:-4]:
                idxi.append(y)
                hdf.append(filters2[i])
                hdfr.append(filters2[i][:filters2[i].find('_MAG')+4]+'ERR'+filters2[i][filters2[i].find('_MAG')+4:])
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
        mag = dataF[hdf[i]][index]
        magerr = dataF[hdfr[i]][index]
        if mag <= 100 and magerr < 10 and mag>0:
            MAG.append(mag)
            MAGERR.append(magerr)
            w.append(wavelength[i])
            wr.append(width[i])
            hdf2.append(hdf[i])
    return sort([MAG,MAGERR,w,wr,hdf2])






NBfdata = np.loadtxt('.\\Filters\%s' % 'UVISTA_NB118.txt')
NBc = Peak(NBfdata[:,0],NBfdata[:,1])
NBw = FWHM(NBfdata[:,0],NBfdata[:,1])





def Redshift(lamb,cen=NBc): #find redshift of wavelength at central value cen (default NB118)
    return cen/lamb - 1
#lines = np.array(pd.read_csv('Lines.csv')['w'])
#lines = emit
#LineNames = np.array(pd.read_csv('Lines.csv')['line'])

lines = pd.read_csv('MilvangLines.csv')['line']



def InFilter(z,cen=NBc,width=NBw): #Does a given redshift have and emission line at this redshift in filter with peak cen and FWHM width
    z = np.asarray(z)
    out = np.zeros(len(z))
    for i in range(len(lines)):
        Eline = cen - (lines[i]*(1+z)+width/2)
        idx = np.where( (0 <= Eline) & (Eline <= width) | (z<0) )[0]
        out[idx] = 1
    return np.where(out == 0)[0]

def InFilter2(cen=NBc,width=NBw,N_sigma=1): #Does a given redshift have and emission line at this redshift in filter with peak cen and FWHM width
    zl = np.asarray(zphot['zPDF_l68']) - (N_sigma-1) * (zphot['zPDF']-zphot['zPDF_l68'])
    zu = np.asarray(zphot['zPDF_u68']) + (N_sigma-1) * (zphot['zPDF_u68']-zphot['zPDF'])
    out = np.zeros(len(zl))
    out[np.where( (zphot['zPDF'] <= 0) | (zphot['zBEST'] > 10) )[0]] = 1
    for i in range(len(lines)):
        wl = lines[i]*(1+zl)-width/2
        wu = lines[i]*(1+zu)+width/2
        idx = np.where( (cen >= wl) & (cen <= wu) )[0]
        out[idx] += 1
    return np.where(out == 0)[0]



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
        return np.array(iout)

def detected_in_farmer(i_c,i_f):
    a = np.asarray(data['ALPHA_J2000'][i_c] - dataF['ALPHA_J2000'][i_f])
    b = np.asarray(data['DELTA_J2000'][i_c] - dataF['DELTA_J2000'][i_f])
    d = np.sqrt(a**2+b**2)
    return d<1e-4




#%%
JNB,JNBr = corr(np.arange(len(data)),True)
sel = np.where( (data['UVISTA_NB118_MAG'+aper]>0)  
               & (data['HSC_r_MAG'+aper]>0) #& (data['HSC_r_MAG'+aper] > 0 )
               & ( data['UVISTA_NB118_FLUX'+aper ] >= 5*data['UVISTA_NB118_FLUXERR'+aper ] )
               & (zphot['type'] == 0) & (zphot['zBEST'] < 0)
               & (JNB >= 0.2) & (JNB>2.5*JNBr)
               )[0]


rmag,JNB,ID,cutn = sort([data['HSC_r_MAG'+aper][sel],JNB[sel],data['ID'][sel],sel])#,direction='asc')



plt.close(fig=8)
plt.figure(8,figsize=(6,4),dpi=100)
ax = plt.subplot(111)
plt.hist(JNB,200,range=(0,np.max(JNB)),fc='k')
# print('%i Sources'%len(JNB))
ax.set_yscale('log')
plt.ylabel('Number of sources')
plt.xlabel('$(J-NB118)_{corr}$ mag')
plt.xlim([0,np.max(JNB)])
plt.tight_layout()


cutn2 = []
sortoff = [0,1,5,6,9,10,12,14,16,17,21,24,25,26,27,31,36,37,38,40,46,48]
for i in range(len(cutn)):
    if i not in sortoff:
        cutn2.append(cutn[i])
cutn = np.asarray(cutn2)


cutn2 = []
sortoff = [28,29,30,31,32,34,37,38,40,41,45,46,48]
for i in range(len(cutn)):
    if i not in sortoff:
        cutn2.append(cutn[i])
cutn = np.asarray(cutn2)


cutn2 = []
sortoff = [27,36,37,38,40,42,44,48]
for i in range(len(cutn)):
    if i not in sortoff:
        cutn2.append(cutn[i])
cutn = np.asarray(cutn2)


cutn2 = []
sortoff = [41,43,45,48,52,53,54,55,59,61]
for i in range(len(cutn)):
    if i not in sortoff:
        cutn2.append(cutn[i])
cutn = np.asarray(cutn2)


print('%i Sources'%len(cutn))
cutn = cutn[:49]

#%% #Making a huge plot from cutn


Size = xysize(cutn)
Ra = data['ALPHA_J2000'][cutn]
Dec = data['DELTA_J2000'][cutn]
ID = data['ID'][cutn]

i_FARMER = LocateFarmer(Ra, Dec)
RaF = dataF['ALPHA_J2000'][i_FARMER]
DecF = dataF['DELTA_J2000'][i_FARMER]




# frame = pd.DataFrame({'CLASSIC_ID':ID , 'Ra':Ra , 'Dec':Dec , 
#                       'r_mag':np.asarray(data['HSC_r_MAG_APER2'][cutn]),
#                       'NB118_mag':np.asarray(data['UVISTA_NB118_MAG_APER2'][cutn]),
#                       'Narrow_Excess_Corr':np.asarray(corr(cutn)),})
# frame.to_csv('NB118ExcessData.csv',index=False)


SIZE = np.ceil(np.ones(len(Size))*8*Size)


handles = [plt.Rectangle((0,0),1,.5,color=c) for c in ['green','Gray']]

plt.close(fig=4)
fig = plt.figure(4,figsize=(25,18),dpi=100)
for i in range(len(cutn)):
    axes = plt.subplot(*subidex(len(cutn)),1+i) 
    labels = [r'$r_{%i}=%.3g$' % (i,data['HSC_r_MAG_APER2'][cutn[i]]),'InFarm %s'%detected_in_farmer(cutn[i],i_FARMER[i])]
    plt.legend(handles,labels,loc='upper left')
    ThumbNail2(Ra[i],Dec[i],0,SIZE[i].astype(int),addRA=[Ra[i],RaF[i]],addDEC = [Dec[i],DecF[i]],addSize=[Size[i],Size[i]],subplot=True,axes=axes,filters=['Ks','NB118','Y'])
plt.suptitle('Ks, NB118 and Y')
plt.tight_layout()
fig.subplots_adjust(top=0.96)

plt.close(fig=5)
fig = plt.figure(5,figsize=(25,18),dpi=100)
for i in range(len(cutn)):
    axes = plt.subplot(*subidex(len(cutn)),1+i)
    labels = [r'$r_{%i}=%.3g$' % (i,data['HSC_r_MAG_APER2'][cutn[i]]),'InFarm %s'%detected_in_farmer(cutn[i],i_FARMER[i])]
    plt.legend(handles,labels,loc='upper left')
    ThumbNail2(Ra[i],Dec[i],0,SIZE[i].astype(int),addRA=[Ra[i],RaF[i]],addDEC = [Dec[i],DecF[i]],addSize=[Size[i],Size[i]],subplot=True,axes=axes,filters=['Ks','J','Y'],)
plt.suptitle('Ks, J and Y')
plt.tight_layout()
fig.subplots_adjust(top=0.96)

# plt.close(fig=4)
# fig = plt.figure(4,figsize=(25,18),dpi=100)
# for i in range(len(cutn)):
#     axes = plt.subplot(*subidex(len(cutn)),1+i) 
#     labels = [r'$r_{%i}=%.3g$' % (i,data['HSC_r_MAG_APER2'][cutn[i]])]
#     plt.legend(handles,labels,loc='upper left')
#     ThumbNail2(Ra[i],Dec[i],Size[i],SIZE[i].astype(int),subplot=True,axes=axes,filters=['Y','NB118','Ks'],sigma=0,ff=[1,1,1])
# plt.suptitle('Y, NB118 and Ks')
# plt.tight_layout()
# fig.subplots_adjust(top=0.96)

# plt.close(fig=5)
# fig = plt.figure(5,figsize=(25,18),dpi=100)
# for i in range(len(cutn)):
#     axes = plt.subplot(*subidex(len(cutn)),1+i)
#     labels = [r'$r_{%i}=%.3g$' % (i,data['HSC_r_MAG_APER2'][cutn[i]])]
#     plt.legend(handles,labels,loc='upper left')
#     ThumbNail2(Ra[i],Dec[i],Size[i],SIZE[i].astype(int),subplot=True,axes=axes,filters=['Y','J','Ks'],sigma=0,ff=[1,1,1])
# plt.suptitle('Y, J and Ks')
# plt.tight_layout()
# fig.subplots_adjust(top=0.96)


#%% Scatter of emission liners

#scatter / 2d histogram over objekter ved en emissions linje
#4 plots som scatter, fave som rødforskydninger


#Hardest emitters (ingen specz)
#Igen emission ved den rødforskydning



#Kasse med fleste emitter ved 0.8, 1.4 og 2.2
#histogrammer ud fra de tre rødfoskydninger 
def wmean(x,sx):
    w = 1/sx**2
    m = np.sum(x*w)/np.sum(w)
    sm = 1/(np.sqrt(np.sum(w)))
    return m,sm


JNB,JNBr = corr(np.arange(len(data)),True)

cutn = np.where( (JNB > .2) & (JNB > 2.5*JNBr)
                & (zphot2['z_spec'] < 0) #& (zphot['type'] == 0) 
                & (zphot['zBEST'] > 0) & (zphot['zBEST'] <= 10)
                & (data['FLAG_HSC'] == 0 ) & (data['FLAG_SUPCAM'] == 0 )
                & (data['UVISTA_NB118_FLUX'+aper ] >= 5*data['UVISTA_NB118_FLUXERR'+aper ])
                & (data['UVISTA_J_FLUX'+aper ] >= 3*data['UVISTA_J_FLUXERR'+aper ])
                & (data['UVISTA_NB118_FLAGS'] == 0)
                
                )[0]

Fcut = np.where( (zphot2F['z_spec'] < 0)
                & (dataF['FLAG_HSC'] == 0 ) & (dataF['FLAG_SUPCAM'] == 0 )
                )[0]




Ra = data['ALPHA_J2000'][cutn]
Dec = data['DELTA_J2000'][cutn]



Fcut = Fcut[LocateFarmer(Ra, Dec,Fcut)]

sel = detected_in_farmer(cutn, Fcut)

Fcut = np.delete(Fcut[sel],169)
cutn = np.delete(cutn[sel],169)



Czl = np.array(zphot['zBEST'][cutn])
Cze = np.array(zphot2['z_phot'][cutn])
Ra = np.array(data['ALPHA_J2000'][cutn])
Dec = np.array(data['DELTA_J2000'][cutn])
Fzl = np.array(zphotF['zBEST'][Fcut])
Fze = np.array(zphot2F['z_phot'][Fcut])
RaF = np.array(dataF['ALPHA_J2000'][Fcut])
DecF = np.array(dataF['DELTA_J2000'][Fcut])
IDC = np.array(data['ID'][cutn])
IDF = np.array(data['ID'][Fcut])
JNB,JNBr = corr(cutn,True)

Z = np.zeros([4,len(Czl)])
Zr = np.copy(Z)
Z[:2] = Czl,Cze
Z[2:] = Fzl,Fze

Zr[0] = ( zphot['zPDF_u68'][cutn]-zphot['zBEST'][cutn] + zphot['zPDF_u68'][cutn]-zphot['zBEST'][cutn] ) / 2
Zr[1] = ( zphot2['z_min_risk'][cutn]-zphot2['z160'][cutn] + zphot2['z840'][cutn]-zphot2['z_min_risk'][cutn] ) / 2
Zr[2] = ( zphotF['zPDF_u68'][Fcut]-zphotF['zBEST'][Fcut] + zphotF['zPDF_u68'][Fcut]-zphotF['zBEST'][Fcut] ) / 2
Zr[3] = ( zphot2F['z_min_risk'][Fcut]-zphot2F['z160'][Fcut] + zphot2F['z840'][Fcut]-zphot2F['z_min_risk'][Fcut] ) / 2

z = np.zeros(len(Czl))
zr = np.zeros(len(Czl))
for i in range(len(Czl)):
    z[i],zr[i] = wmean(Z[:,i],Zr[:,i])

#%%


plt.close('all')

R0 = 150.3
D0 = 2.8
Bue = 7/60
cmap = 'hsv'
marker = 'o'
lw = .005
anno = "7'"





# ci = nearest_index(Ra,150.4125)
# ThumbNail2(Ra[ci],Dec[ci],0,2000,addRA=Ra,addDEC=Dec,addSize=xysize(cutn))

def InFilter3(z_search,z,zr,cen=NBc,width=NBw,N_sigma=1): #Does a given redshift have and emission line at this redshift in filter with peak cen and FWHM width
    lines = cen/(1+z_search)
    out = np.zeros(len(z))
    wl = lines*(1+z-zr*N_sigma)-width/2
    wu = lines*(1+z+zr*N_sigma)+width/2
    idx = np.where( (cen >= wl) & (cen <= wu) )[0]
    out[idx] += 1
    return np.where(out > 0)[0]

plt.figure(figsize=(7,4),dpi=100)
ax = plt.subplot(111)

s1 = InFilter3(0.814443,z,zr,N_sigma=3)
s2 = InFilter3(1.401256,z,zr,N_sigma=3)
s3 = InFilter3(2.194760,z,zr,N_sigma=3)

plt.hist(z,100,fc='k',range=(0,5))
plt.hist(z[s1],100,fc='b',range=(0,5))
plt.hist(z[s2],100,fc='g',range=(0,5))
plt.hist(z[s3],100,fc='r',range=(0,5))

handles = [plt.Rectangle((0,0),1,.5,color=c) for c in ['black','blue','green','red']]
labels = ['All',r'z = 0.8 (H$\alpha$)',r'z = 1.4 (OIII)',r'z = 2.2 (OII)']
plt.legend(handles,labels,loc='upper right')
plt.xlabel('Photometric redshift (combined)')
plt.ylabel('Number of sources')
plt.tight_layout()


plt.figure(figsize=(8,8),dpi=80)

ax = plt.subplot(111)
plt.gca().invert_xaxis()
plt.scatter(Ra[s1],Dec[s1],fc='b',marker=marker,lw=lw,label=r'z = 0.8 (H$\alpha$)')
plt.scatter(Ra[s2],Dec[s2],fc='g',marker=marker,lw=lw,label=r'z = 1.4 (OIII)')
plt.scatter(Ra[s3],Dec[s3],fc='r',marker=marker,lw=lw,label=r'z = 2.2 (OII)')
plt.legend()
plt.plot([R0-Bue/2,R0+Bue/2],[D0]*2,'k',lw=3)
plt.plot([R0+Bue/2]*2,[D0,D0-Bue],'k',lw=3)
ax.annotate(anno,xy=(R0,D0-0.03))
plt.tight_layout()



N = 1000
X = np.linspace(min(Ra),max(Ra),N)
Y = np.linspace(min(Dec),max(Dec),N)

def Zf(x,y,s,xw=7.5/60,yw=4/60):
    out = np.zeros([len(x),len(y)])
    for ix in range(len(x)):
        for iy in range(len(y)):
            out[ix,iy] = len(np.where( (np.abs(Ra[s]-x[ix]) <= xw/2) & (np.abs(Dec[s]-y[iy]) <= yw/2) ) [0])
            #out[ix,iy] = np.sum(Bue/2 >= np.sqrt( (Ra[s]-x[ix])**2 + (Dec[s]-y[iy])**2 ) )
    return out
if 'amount1' not in locals():
    amount1 = np.rot90(Zf(X,Y,s1))
    amount2 = np.rot90(Zf(X,Y,s2))
    amount3 = np.rot90(Zf(X,Y,s3))
    y1,x1 = np.where(np.max(amount1) == amount1)#np.unravel_index(np.argmax(amount1),amount1.shape)
    y2,x2 = np.where(np.max(amount2) == amount2)
    y3,x3 = np.where(np.max(amount3) == amount3)

S1 = s1[np.where( (np.abs(np.median(X[x1])-Ra[s1]) <= Bue/2) & (np.abs(np.median(Y[::-1][y1])-Dec[s1]) <= Bue/2) )[0]]
S2 = s2[np.where( (np.abs(np.median(X[x2])-Ra[s2]) <= Bue/2) & (np.abs(np.median(Y[::-1][y2])-Dec[s2]) <= Bue/2) )[0]]
S3 = s3[np.where( (np.abs(np.median(X[x3])-Ra[s3]) <= Bue/2) & (np.abs(np.median(Y[::-1][y3])-Dec[s3]) <= Bue/2) )[0]]

plt.plot(Ra[S1],Dec[S1],'ok',alpha=.4)
plt.plot(Ra[S2],Dec[S2],'ok',alpha=.4)
plt.plot(Ra[S3],Dec[S3],'ok',alpha=.4)


def box(x,y,size=Bue,style='k',alpha=0.1,Wx=7.5/60,Wy=4/60):
    wx = Wx/2 ; wy = Wy/2
    plt.plot([x-wx,x+wx],[y+wy]*2,style,alpha=alpha)
    plt.plot([x-wx,x+wx],[y-wy]*2,style,alpha=alpha)
    plt.plot([x-wx]*2,[y-wy,y+wy],style,alpha=alpha)
    plt.plot([x+wx]*2,[y-wy,y+wy],style,alpha=alpha)




box(np.median(X[x1]),np.median(Y[::-1][y1]),style='k')
box(np.median(X[x2]),np.median(Y[::-1][y2]),style='k')
box(np.median(X[x3]),np.median(Y[::-1][y3]),style='k')


fl1 = np.array([False]*len(z))
fl2 = np.array([False]*len(z))
fl3 = np.array([False]*len(z))
fl1[s1] = True
fl2[s2] = True
fl3[s3] = True
nof = np.array([False]*len(z))
for i in range(len(z)):
    if fl1[i] == False and fl2[i] == False and fl3[i] == False:
        nof[i] = True

df = pd.DataFrame({'Ra':Ra , 'Dec':Dec , 'ID_CLASSIC':IDC , 'ID_FARMER':IDF , 'z_mean':z , 'sigma_z_mean':zr
                    ,'zCLASSIC_lp':Czl , 'zCLASSIC_ez':Cze,'zFARMER_lp':Fzl , 'zFARMER_ez':Fze
                    ,'NB118_MAG':data['UVISTA_NB118_MAG_APER2'][cutn],'JNB':JNB , 'sigma_JNB':JNBr
                    ,'in_z=0.8':fl1 ,'in_z=1.4':fl2 ,'in_z=2.2':fl3 , 'not_flagged':nof
                    })
# df.to_csv('ObservationSelection.csv',index=False)
df2 = pd.read_csv('ObservationSelection.csv')




plt.figure(figsize=(18,6),dpi=80)
extent = [min(Ra),max(Ra),min(Dec),max(Dec)]
ax = plt.subplot(131)
plt.imshow(amount1,vmin=2,extent=extent,cmap='Blues')
plt.scatter(X[x1],Y[::-1][y1],fc='r')
plt.gca().invert_xaxis()
plt.title(r'z = 0.8 (H$\alpha$)')

ax = plt.subplot(132)
plt.imshow(amount2,vmin=2,extent=extent,cmap='Greens')
plt.scatter(X[x2],Y[::-1][y2],fc='r')
plt.gca().invert_xaxis()
plt.title(r'z = 1.4 (OIII)')

ax = plt.subplot(133)
plt.imshow(amount3,vmin=2,extent=extent,cmap='Reds')
plt.scatter(X[x3],Y[::-1][y3],fc='r')
plt.gca().invert_xaxis()
plt.title(r'z = 2.2 (OII)')

plt.tight_layout()


#%%

def rotate(ox,oy, px,py, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return qx, qy

def rectang(X,Y,rot,fmt='k',alpha=0.1,label=None,Wx=7.5/60,Wy=4/60):
    wx = Wx/2 ; wy = Wy/2
    A = rotate(X,Y,X-wx,Y+wy,rot)
    B = rotate(X,Y,X+wx,Y+wy,rot)
    C = rotate(X,Y,X+wx,Y-wy,rot)
    D = rotate(X,Y,X-wx,Y-wy,rot)
    plt.plot([A[0],B[0]],[A[1],B[1]],fmt,alpha=alpha,label=label)
    plt.plot([B[0],C[0]],[B[1],C[1]],fmt,alpha=alpha)
    plt.plot([C[0],D[0]],[C[1],D[1]],fmt,alpha=alpha)
    plt.plot([D[0],A[0]],[D[1],A[1]],fmt,alpha=alpha)

def Heron(a,b,c):
    s = (a+b+c)/2
    return np.sqrt( s * (s-a) * (s-b) * (s-c) )
def PD(A,B): #Distance between two points
    x1,y1 = A
    x2,y2 = B
    return np.sqrt((x1-x2)**2+(y1-y2)**2)



def insideRect(Ra,Dec,X=0,Y=0,rot=0,Wx=7.5/60,Wy=4/60): #Is Ra,Dec insice rectangle at X,Y rotated with angle rot
    Ra = np.asarray(Ra)
    Dec = np.asarray(Dec)
    wx = 0.5*Wx 
    wy = 0.5*Wy
    P = np.array([Ra,Dec])
    A = np.array(rotate(X,Y,X-wx,Y+wy,rot))
    B = np.array(rotate(X,Y,X+wx,Y+wy,rot))
    C = np.array(rotate(X,Y,X+wx,Y-wy,rot))
    d1 = np.asarray(np.dot([B[0]-A[0],B[1]-A[1]],[P[0]-A[0],P[1]-A[1]]))
    d2 = np.asarray(np.dot([B[0]-A[0],B[1]-A[1]],[B[0]-A[0],B[1]-A[1]]))
    d3 = np.asarray(np.dot([C[0]-B[0],C[1]-B[1]],[P[0]-B[0],P[1]-B[1]]))
    d4 = np.asarray(np.dot([C[0]-B[0],C[1]-B[1]],[C[0]-B[0],C[1]-B[1]]))
    out = np.array([False]*len(Ra))
    out[np.where( (0 <= d1) & (d1 <= d2) & (0 <= d3) & (d3 <= d4) )[0]] = True
    return out


rot = np.linspace(0,180,30)/360 *2*np.pi



Ra3 = Ra[s3]
Dec3 = Dec[s3]



if 'Score' not in locals():
    N = 500
    X = np.linspace(min(Ra3),max(Ra3),N)
    Y = np.linspace(min(Dec3),max(Dec3),N)
    
    Skip = np.zeros([len(X),len(Y)])
    for ix in range(len(X)):
        for iy in range(len(Y)):
            if np.min(np.sqrt( (Ra3-X[ix])**2 + (Dec3-Y[iy])**2 )) > 0.5*7.5/60:
                Skip[ix,iy] = 1
    
    
    Score = np.zeros([len(X),len(Y)])
    Rot = np.copy(Score)
    
    for ix in range(len(X)):
        for iy in range(len(Y)):
            if Skip[ix,iy] == 0:
                S = np.zeros(len(rot))
                for i,r in enumerate(rot):
                    S[i] = np.sum(insideRect(Ra3,Dec3,X[ix],Y[iy],rot=r))
                Score[ix,iy] = np.max(S)
                Rot[ix,iy] = rot[np.argmax(S)]
        
plt.close(fig=4)
extent = [min(Ra3),max(Ra3),min(Dec3),max(Dec3)]
plt.figure(4,figsize=(10,8),dpi=100)
plt.imshow(np.rot90(Score),extent=extent,alpha=0.9)
plt.gca().invert_xaxis()
plt.colorbar()
plt.scatter(Ra3,Dec3,fc='r',alpha=0.2)
plt.tight_layout()



x3 = []
y3 = []
Smax = np.max(Score)
r3 = []
for ix in range(len(X)):
        for iy in range(len(Y)):
            if Score[ix,iy] == Smax:
                x3.append(ix),y3.append(iy),r3.append(Rot[ix,iy])
x3 = x3[len(x3)//2]
y3 = y3[len(y3)//2]
r3 = r3[len(r3)//2]


                
# plt.close(fig=5)
# plt.figure(5,figsize=(8,8),dpi=80)

plt.scatter(Ra3,Dec3,fc='r',marker=marker,lw=lw,label=r'z = 2.2 (OII)')

rectang(X[x3],Y[y3],Rot[x3,y3],alpha=0.9)
plt.legend()

#%%
Czl[Czl>2.2] = 2.2
Cze[Cze>2.2] = 2.2
Fzl[Fzl>2.2] = 2.2
Fze[Fze>2.2] = 2.2

plt.figure(figsize=(20,14),dpi=80)


ax = plt.subplot(221)
plt.gca().invert_xaxis()
plt.title('CLASSIC ')
plt.scatter(Ra,Dec,marker=marker,c=Czl,cmap=cmap,lw=lw)
plt.colorbar()
plt.plot([R0-Bue/2,R0+Bue/2],[D0]*2,'k',lw=3)
ax.annotate(anno,xy=(R0,D0-0.04))
plt.title('CLASSIC LePhare')


ax = plt.subplot(222)
plt.gca().invert_xaxis()
plt.scatter(Ra[Cze >=0],Dec[Cze >=0],marker=marker,c=Cze[Cze >=0],cmap=cmap,lw=lw)
plt.colorbar()
plt.plot([R0-Bue/2,R0+Bue/2],[D0]*2,'k',lw=3)
ax.annotate(anno,xy=(R0,D0-0.04))
plt.title('CLASSIC EAZY')


ax = plt.subplot(223)
plt.gca().invert_xaxis()
plt.scatter(RaF,DecF,marker=marker,c=Fzl,cmap=cmap,lw=lw)
plt.colorbar()
plt.plot([R0-Bue/2,R0+Bue/2],[D0]*2,'k',lw=3)
ax.annotate(anno,xy=(R0,D0-0.04))
plt.title('FARMER LePhare')


ax = plt.subplot(224)
plt.gca().invert_xaxis()
plt.scatter(RaF[Fze >=0],DecF[Fze >=0],marker=marker,c=Fze[Fze >=0],cmap=cmap,lw=lw)
plt.colorbar()
plt.plot([R0-Bue/2,R0+Bue/2],[D0]*2,'k',lw=3)
ax.annotate(anno,xy=(R0,D0-0.04))
plt.title('FARMER EAZY')

plt.tight_layout()
#%%

plt.close('all')
JNB,JNBr = corr(np.arange(len(data)),True)
cutn = np.where( (zphot2['z_spec'] < 0) 
                & ( JNB > 0.2) & ( JNB < 50) & ( JNB > 2.5*JNBr )
                & (data['FLAG_HSC'] == 0 ) & (data['FLAG_SUPCAM'] == 0 )
                & (data['UVISTA_NB118_FLUX'+aper ] >= 5*data['UVISTA_NB118_FLUXERR'+aper ])
                & (data['UVISTA_J_FLUX'+aper ] >= 3*data['UVISTA_J_FLUXERR'+aper ])
                & (data['UVISTA_NB118_FLAGS'] == 0)
                )[0]
JNB,JNBr,cutn = sort([JNB[cutn],JNBr[cutn],cutn],direction='desc')
plt.figure()
plt.hist(JNB,100,fc='k')
plt.xlabel(r'$(J-NB118)_{corr}$')
plt.ylabel('Number of sources')


Size = xysize(cutn)
Ra = data['ALPHA_J2000'][cutn]
Dec = data['DELTA_J2000'][cutn]
ID = data['ID'][cutn]
i_FARMER = LocateFarmer(Ra, Dec)
RaF = dataF['ALPHA_J2000'][i_FARMER]
DecF = dataF['DELTA_J2000'][i_FARMER]





#%%  Lensed Plot Selection
plt.close('all')


idx = InFilter2(N_sigma=2)

z = zphot['zPDF'][idx]




idx = idx[np.where(   (data['UVISTA_NB118_FLAGS'][idx] == 0)
                & ( data['UVISTA_NB118_FLUX'+aper ][idx] >= 5*data['UVISTA_NB118_FLUXERR'+aper ][idx] )
                & ( data['UVISTA_J_FLUX'+aper ][idx] >= 3*data['UVISTA_J_FLUXERR'+aper ][idx] )
                &  (data['FLAG_HSC'][idx] == 0 ) &  (data['FLAG_SUPCAM'][idx] == 0 )
                & (zphot2['z_spec'][idx] < 0)
                )[0]]




z2 = zphot['zPDF'][idx]
handles = [plt.Rectangle((0,0),1,.5,color=c) for c in ['black','gray','red']]



plt.figure(2,figsize=(8,3),dpi=100)
ax = plt.subplot(111)
plt.hist(zphot['zPDF'],500,range=(0,10),fc='k',zorder=0,alpha=1)
plt.hist(z,             500,range=(0,10),fc='gray',zorder=1,alpha=1)
plt.hist(z2,            500,range=(0,10),fc='r',zorder=2,alpha=1)

print('%i Sources not in filter'%len(z))
print('%i Detected in sigma criteria'%len(z2))
for i in range(len(lines)):
    l1 = plt.plot(np.ones(2)*Redshift(lines[i]),[0,20000],'-g',alpha=0.5)
handles.append(l1[0])
plt.xlim([0,10])
plt.ylim([.6,20000])
ticks = [i for i in range(11)]
labels = ['%.10g'%i for i in ticks]
plt.xticks(ticks,labels)
ax.set_yscale('log')
plt.legend(handles,['All sources','Emission line sources removed','5$\sigma$ NB118 and 3$\sigma$ J','Emission lines'],loc='upper right')
plt.xlabel('Redshift')
plt.ylabel('Number of sources')
plt.tight_layout()




idx = idx[corr(idx) >= -2]
JNB, JNBr = corr(idx,True)
NB118 = data['UVISTA_NB118_MAG'+aper ][idx]
NBr = data['UVISTA_NB118_MAGERR'+aper ][idx]


plt.figure(10,figsize=(6,4),dpi=100)
ax = plt.subplot(111)
plt.hist(JNB,200,range=(0,3.5))
# print('%i Sources'%len(JNB))
ax.set_yscale('log')
plt.ylabel('Number of sources')
plt.xlabel('$(J-NB118)_{corr}$ mag')
plt.xlim([0,3.5])
plt.tight_layout()



plt.figure(1,figsize=(6,4),dpi=100)
plt.hist2d(NB118,JNB,200,norm=LogNorm(vmin=1))
plt.colorbar()
plt.xlabel('$NB118$ mag')
plt.ylabel('$(J-NB118)_{corr}$ mag')
plt.tight_layout()



JNBcut = 0.2
SigmaJNB = 2.5


idx = np.intersect1d( idx[corr(idx) > JNBcut] , idx[JNB > SigmaJNB*JNBr] )
print('%i have narrowband excess'%len(idx))
idx = np.concatenate((idx,[1389194, 1216956]))
JNB, JNBr = corr(idx,True)
NB118 = data['UVISTA_NB118_MAG'+aper ][idx]
NBr = data['UVISTA_NB118_MAGERR'+aper ][idx]




plt.figure(3,figsize=(8,5),dpi=100)
plt.errorbar(NB118,JNB,JNBr,NBr,ms=2,capsize=2,lw=1,fmt='.k')
plt.xlabel('$NB118$ mag')
plt.ylabel('$(J-NB118)_{corr}$ mag')
plt.tight_layout()


cutn = np.copy(idx)
Size = xysize(cutn)
Ra = data['ALPHA_J2000'][cutn]
Dec = data['DELTA_J2000'][cutn]
ID = data['ID'][cutn]

# IDadd = [380849,243863]
# CutAdd = np.intersect1d(zphot['Id'],IDadd,return_indices=True)[1]
# cutn = np.append(cutn,CutAdd)

dist = np.zeros([len(lines),len(cutn)])
for i,z in enumerate(Redshift(lines)):
    dist[i] = np.abs(zphot['zPDF'][cutn]-z)
dist = np.min(dist,axis=0)

dist,JNB,cutn = sort([dist,JNB,cutn],'desc')
# cutn = cutn[:49]

index = [8,9,10,11,14,21,23,28,29,30,61,96,99,107,126,135,139,141,144]
Reason = ['Very strong line'
          ,'LePhare fit far off'
          ,'Two objects (Lense?)','Two objects (Lense?)'
          ,'Well defined shape, only one fit'
          ,'Lots of lines'
          ,'Lots of lines'
          ,'Strong Lines'
          ,'Strong Lines'
          ,'Strong Lines'
          ,'Strong Narrowband'
          ,'Very broad emission line?'
          ,'Possible many lines, or error/variable?'
          ,'Many lines? or error'
          ,'Huge variance between EAZY and LP'
          ,'Two objects'
          ,'Two objects at same redshift? (fits are very consistent)'
          ,'Strong narrowband excess, very inconsistent redshifts'
          ,'Strong narrowband, varying fits']






cutnold = np.copy(cutn)
#%% #Making a huge plot from cutn
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

cutn = cutnold[[8,12,9,81]]

Size = xysize(cutn)
Ra = data['ALPHA_J2000'][cutn]
Dec = data['DELTA_J2000'][cutn]
ID = data['ID'][cutn]
i_FARMER = LocateFarmer(Ra, Dec)
RaF = dataF['ALPHA_J2000'][i_FARMER]
DecF = dataF['DELTA_J2000'][i_FARMER]


frame = pd.DataFrame({'ID':ID , 'Ra':Ra , 'Dec':Dec})
frame.to_csv('.//LensedPlots/LensedPlots.csv',index=False)

#spec_z 
#højeste equivalent width objekter



SIZE = np.ceil(np.ones(len(Size))*8*np.abs(Size))


handles = [plt.Rectangle((0,0),1,.5,color=c) for c in ['green','Gray']]


plt.close(fig=4)
fig = plt.figure(4,figsize=(3,3),dpi=200)
for i in range(len(cutn)):
    axes = plt.subplot(*subidex(len(cutn)),1+i) 
    labels = [r'$JNB_{%i}=%.3g$' % (i,JNB[i]),'InFarm %s'%detected_in_farmer(cutn[i],i_FARMER[i])]
    #plt.legend(handles,labels,loc='upper left')
    ThumbNail2(Ra[i],Dec[i],0,40,addRA=[Ra[i],RaF[i]],addDEC = [Dec[i],DecF[i]],addSize=[Size[i],Size[i]],subplot=True,axes=axes,filters=['Ks','NB118','Y'])
    plt.annotate('LS$_%i$'%(i+1),xy=(0,0),color='white')
    plt.axis('off')
#plt.suptitle('Ks, NB118 and Y')

plt.tight_layout()
fig.subplots_adjust(wspace=0.01,hspace=0.01,right=1,left=0,top=1,bottom=0)

plt.close(fig=5)
fig = plt.figure(5,figsize=(3,3),dpi=200)
for i in range(len(cutn)):
    axes = plt.subplot(*subidex(len(cutn)),1+i)
    labels = [r'$JNB_{%i}=%.3g$' % (i,JNB[i]),'InFarm %s'%detected_in_farmer(cutn[i],i_FARMER[i])]
    #plt.legend(handles,labels,loc='upper left')
    ThumbNail2(Ra[i],Dec[i],0,40,addRA=[Ra[i],RaF[i]],addDEC = [Dec[i],DecF[i]],addSize=[Size[i],Size[i]],subplot=True,axes=axes,filters=['Ks','J','Y'])
    plt.annotate('LS$_%i$'%(i+1),xy=(0,0),color='white')
    plt.axis('off')
#plt.suptitle('Ks, J and Y')

plt.tight_layout()
fig.subplots_adjust(wspace=0.01,hspace=0.01,right=1,left=0,top=1,bottom=0)

# handles = [plt.Rectangle((0,0),1,.5,color=c) for c in ['green','Gray']]



# plt.close(fig=4)
# fig = plt.figure(4,figsize=(22,16),dpi=80)
# for i in range(len(cutn)):
#     axes = plt.subplot(*subidex(len(cutn)),1+i) 
#     labels = [r'$z_{%i}=%.2g$' % (i,zphot['zBEST'][cutn[i]])]
#     plt.legend(handles,labels,loc='upper left')
#     ThumbNail2(Ra[i],Dec[i],Size[i],SIZE[i].astype(int),subplot=True,axes=axes,filters=['Y','NB118','Ks'],sigma=0,ff=[1,1,1])
# plt.suptitle('Y, NB118 and Ks')
# plt.tight_layout()
# fig.subplots_adjust(top=0.96)

# plt.close(fig=5)
# fig = plt.figure(5,figsize=(22,16),dpi=80)
# for i in range(len(cutn)):
#     axes = plt.subplot(*subidex(len(cutn)),1+i)
#     labels = [r'$z_{%i}=%.2g$' % (i,zphot['zBEST'][cutn[i]])]
#     plt.legend(handles,labels,loc='upper left')
#     ThumbNail2(Ra[i],Dec[i],Size[i],SIZE[i].astype(int),subplot=True,axes=axes,filters=['Y','J','Ks'],sigma=0,ff=[1,1,1])
#     #print(Ra[i],Dec[i])
# plt.suptitle('Y, J and Ks')
# plt.tight_layout()
# fig.subplots_adjust(top=0.96)


# ThumbNail(150.5636250,2.4,0,400000,addRa=Ra,addDec=Dec,addSize=Size)





#%%  Selecting objects with no z_phot and z_spec
plt.close('all')
ranged = np.where(  (zphot['zBEST'] < 0)  & (zphot['type'] == 0)
                   & (data['UVISTA_NB118_FLUX_APER2'] >= 5*data['UVISTA_NB118_FLUXERR_APER2']) 
                   & (data['UVISTA_J_FLUX_APER2'] >= 3*data['UVISTA_J_FLUXERR_APER2']) 
                   & (zphot2['z_phot'] < 0) & (zphot2['z_spec'] < 0)
                   & (data['FLAG_HSC'] == 0) & (data['FLAG_SUPCAM']==0)
                   )[0]
Ra = data['ALPHA_J2000'][ranged]
Dec = data['DELTA_J2000'][ranged]

rangedF = np.where((zphotF['zBEST'] < 0)  & (zphotF['type'] == 0)
                   & (zphot2F['z_phot'] < 0)
                   )[0]

closestIdx = rangedF[LocateFarmer(Ra, Dec,rangedF)]

detected1 = detected_in_farmer(ranged, closestIdx)



detected = []
for i in range(len(Ra)):
    if detected1[i] and closestIdx[i] in rangedF:
        detected.append(True)
    else:
        detected.append(False)




waste,cutn,cutnF,detected = sort([data['UVISTA_NB118_FLUX_APER2'][ranged],ranged,closestIdx,detected],'desc')

# cutn =  cutn[:50]
# cutnF= cutnF[:50]

Ra = data['ALPHA_J2000'][cutn]
Dec = data['DELTA_J2000'][cutn]
RaF = dataF['ALPHA_J2000'][cutnF]
DecF = dataF['DELTA_J2000'][cutnF]







ID = data['ID'][cutn]
Size = xysize(cutn)
i_FARMER=cutnF








#%% #Saving specs for galaxies


folder=".//LensedPlots/"
OverWrite = True
Indexing = True

for i in range(len(cutn)):
    if 'Source%iID%i.png' % (i,ID[i]) not in os.listdir(path=folder) or OverWrite:
        fig = plt.figure(7,figsize=(8,9),dpi=60)
        ax1 = plt.subplot(212)
        # MAG,MAGERR,w,wr,hdf2 = Magnitude2(i_FARMER[i])
        # ax1.errorbar(w,MAG,MAGERR,wr,fmt='.b',lw=1,ms=2,capsize=2,label='FARMER')
        MAG,MAGERR,w,wr,hdf2 = Magnitude(cutn[i])
        ax1.errorbar(w,MAG,MAGERR,wr,fmt='.k',lw=1,ms=2,capsize=2,label='CLASSIC')
        # plt.legend()
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
        SIZE = np.int(np.ceil(12*Size[i]))
        SIZE = 40
        
        axes = plt.subplot(221)

        ThumbNail2(Ra[i],Dec[i],0,SIZE,addRA=[Ra[i],RaF[i]],addDEC = [Dec[i],DecF[i]],addSize=[Size[i],Size[i]],subplot=True,axes=axes,filters=['Ks','NB118','Y'])
        
        if Indexing: plt.title('Ks, NB118 and Y     ' + r'ID = %i' % (ID[i]))
        else: plt.title('Ks, NB118 and Y     ' + r'ID = %i   $(J-NB)_{corr}=%.4g \pm %.2g$' % (ID[i],JNB[i],JNBr[i]))
        axes = plt.subplot(222)

        ThumbNail2(Ra[i],Dec[i],0,SIZE,addRA=[Ra[i],RaF[i]],addDEC = [Dec[i],DecF[i]],addSize=[Size[i],Size[i]],subplot=True,axes=axes,filters=['Ks','J','Y'])
        plt.title( 'Ks, J and Y     ' + RaDeStr(Ra[i],Dec[i])[0] +' , '+ RaDeStr(Ra[i],Dec[i])[1] )
        
        supplot = r'CLASSIC:   $z_{LePhare}=%.4g_{-%.2g}^{+%.2g}$  ,  $z_{EAZY}=%.4g_{-%.2g}^{+%.2g}$' % (zphot['zPDF'][cutn[i]]
                 ,zphot['zPDF'][cutn[i]]-zphot['zPDF_l68'][cutn[i]],zphot['zPDF_u68'][cutn[i]]-zphot['zPDF'][cutn[i]]
                 ,zphot2['z500'][cutn[i]],zphot2['z500'][cutn[i]]-zphot2['z160'][cutn[i]]
                 ,zphot2['z840'][cutn[i]]-zphot2['z500'][cutn[i]]
                 )
        supplot2= r'FARMER:    $z_{LePhare}=%.4g_{-%.2g}^{+%.2g}$  ,  $z_{EAZY}=%.4g_{-%.2g}^{+%.2g}$' % (zphotF['zPDF'][i_FARMER[i]]
                 ,zphotF['zPDF'][i_FARMER[i]]-zphotF['zPDF_l68'][i_FARMER[i]],zphotF['zPDF_u68'][i_FARMER[i]]-zphotF['zPDF'][i_FARMER[i]]
                 ,zphot2F['z500'][i_FARMER[i]],zphot2F['z500'][i_FARMER[i]]-zphot2F['z160'][i_FARMER[i]]
                 ,zphot2F['z840'][i_FARMER[i]]-zphot2F['z500'][i_FARMER[i]]
                 )
        
        fig.subplots_adjust(top=0.88)
        if detected_in_farmer(cutn[i], i_FARMER[i]):
            plt.suptitle(supplot+'\n'+supplot2,fontsize=20)
        else:
            plt.suptitle(supplot,fontsize=20)
        plt.tight_layout()
        plt.savefig(folder+'Source%iID%i.png' % (i,ID[i]),dpi=400)
        plt.close(fig=7)
        if len(cutn) > 10 and i % (len(cutn)//10) == 0:
            print('%.3g%% done' % (i/len(cutn)*100))




# df = pd.DataFrame({'ID':ID , 'Ra':Ra , 'Dec':Dec , 'J-NB':JNB , 'sigma JNB':JNBr})
# df.to_csv('Nb118EmittersInfo.csv',index=False)















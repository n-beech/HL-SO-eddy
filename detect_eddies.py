import numpy as np
import xarray as xr
import datetime as dt
import sys
import xgcm
from joblib import Parallel, delayed
import eddytools as et
import pickle
import calendar
import os

syear=int(sys.argv[1])
depthin=int(sys.argv[2])
if syear ==1951:
    years=[1951,1952,1953,1954,1955,1956]
if syear==2091:
    years=[2091,2092,2093,2094,2095,2096]
    
depths=['-2.5m','-27.5m','-97.5m','-190.0m']
depth=depths[depthin]

datapath='/PATH/TO/UV/DATA/'
std=np.load('/PATH/TO/STD/DATA/OW_std_'+str(depth)+'_impfil10-100_1951-1956.npy')
bathyreg=np.load('/PATH/TO/BATHYMETRY/DATA/bathy.npy')
oceanreg=np.load('/PATH/TO/SEA/LAND/MASK/omask.npy')

#define the metrics
metrics = {('X'): ['dxC', 'dxG', 'dxF', 'dxV'], # X distances
        ('Y'): ['dyC', 'dyG', 'dyF', 'dyU']}

def define_grid(gridtype='reg',bounds=[-180,180,-80,90],dx=1,dy=1,periodic=True):
    left,right,bottom,top=bounds
    #variables we want to keep
    global lons_c, lats_c, lons_gl, lats_gl, lons_gr, lats_gr, lats_g, \
    vpointslonl, vpointslatl, vpointslonr, vpointslatr, upointslonl, upointslatl, upointslonr, \
    upointslatr, cpointslon, cpointslat, fpointslonl, fpointslatl, fpointslonr, fpointslatr, \
    dxF, dyF, dxC, dyC, dxG, dyG, dxV, dyU 
    
    if gridtype=='reg':
        # the x center and f points on either side
        lons_c = np.arange(left+(0.5*dx), right, dx)
        lons_gl = np.arange(left, right, dx)
        lons_gr = lons_gl+(dx)
        
        lats_c = np.arange(bottom+(0.5*dy), top, dy)
        lats_gl = np.arange(bottom, top, dy)
        lats_gr = lats_gl+(dy)
        
    elif gridtype=='iso':
        
        # the x center and f points on either side
        lons_c = np.arange(left+(0.5*dx), right, dx)
        lons_gl = np.arange(left, right, dx)
        lons_gr = lons_gl+(dx)
        
        #the y 
        lats_g=[bottom]

        #we need one extra center lat point in order to differentiate the correct number of v points
        while lats_g[-1] < top+(dy*(np.cos(np.radians(top)))):
            lats_g.append(lats_g[-1]+dy*(np.cos(np.radians(lats_g[-1]))))

        lats_gl=lats_g[:-1]
        lats_gr=lats_g[1:]
        lats_c=(np.asarray(lats_gl)+np.asarray(lats_gr))/2

        lats_g,lats_gl,lats_gr=lats_g[:-1],lats_gl[:-1],lats_gr[:-1]
        
    else:
        print('grid type not prepared with this function')
        pass
        
    #latlon points in 2d
    vpointslonl, vpointslatl = np.meshgrid(lons_c,lats_gl)
    vpointslonr, vpointslatr = np.meshgrid(lons_c,lats_gr)

    upointslonl, upointslatl = np.meshgrid(lons_gl,lats_c)
    upointslonr, upointslatr = np.meshgrid(lons_gr,lats_c)

    cpointslon, cpointslat = np.meshgrid(lons_c,lats_c)

    fpointslonl, fpointslatl = np.meshgrid(lons_gl,lats_gl)
    fpointslonr, fpointslatr = np.meshgrid(lons_gr,lats_gr)
    
    #distances
    dxC = et.tracking.get_distance_latlon2m(cpointslon[:-1,1:],cpointslat[:-1,1:], 
                                            cpointslon[:-1,:-1],cpointslat[:-1,:-1])
    
    #we use the left lat point because we lose the top row 
    dxV = et.tracking.get_distance_latlon2m(vpointslonl[:,1:],vpointslatl[:,1:], 
                                            vpointslonl[:,:-1],vpointslatl[:,:-1])
    
    #for dxC and dxV we need to manually add the periodic distance
    if periodic:
        endcolumn=et.tracking.get_distance_latlon2m(cpointslon[:-1,0],cpointslat[:-1,0],
                                                    cpointslon[:-1,-1],cpointslat[:-1,-1])
        dxC = np.append(dxC, endcolumn.reshape((endcolumn.shape[0],1)),axis=1)
        endcolumn=et.tracking.get_distance_latlon2m(vpointslonl[:,0],vpointslatl[:,0],
                                                    vpointslonl[:,-1],vpointslatl[:,-1])
        dxV = np.append(dxV, endcolumn.reshape((endcolumn.shape[0],1)),axis=1)

    #for dyC and dyU, we lose the top row that we added above
    #*** to do: change to lose the bottom row, so that this works better in the northern hemisphere
    #the bottom row can be on land over antarctica
    dyC = et.tracking.get_distance_latlon2m(cpointslon[1:,:],cpointslat[1:,:],
                                            cpointslon[:-1,:],cpointslat[:-1,:])
    dyU = et.tracking.get_distance_latlon2m(upointslonl[1:,:],upointslatl[1:,:],
                                            upointslonl[:-1,:],upointslatl[:-1,:])

    #dxG and dyG
    dxG = et.tracking.get_distance_latlon2m(fpointslonr,fpointslatl,fpointslonl,fpointslatl)
    dyG = et.tracking.get_distance_latlon2m(fpointslonl,fpointslatr,fpointslonl,fpointslatl)

    #dxF and dyF. for dxF we have to manually remove the top layer
    dxF = et.tracking.get_distance_latlon2m(upointslonr,upointslatl,upointslonl,upointslatl)
    dxF = dxF[:-1,:]
    dyF = et.tracking.get_distance_latlon2m(vpointslonl,vpointslatr,vpointslonl,vpointslatl)
    
    lats_c=lats_c[:-1]
    cpointslat=cpointslat[:-1,:]
    upointslonl=upointslonl[:-1,:]
    upointslatl=upointslatl[:-1,:]
    upointslonr=upointslonr[:-1,:]
    upointslatr=upointslatr[:-1,:]

define_grid('iso',[-180,180,-80,-40],dx=0.05,dy=0.05, periodic=True)

#MITGCM option has been modified to support FESOM data on the C grid.
detection_parameters = {'model': 'MITgcm',
                    'grid': 'latlon',
                    'hemi': 'south',
                    'start_time': '1951-01-02', # time range start
                    'end_time': '1951-01-01', # time range end
                    'calendar': 'standard', # calendar, must be either 360_day or standard
                    'lon1': -180, # minimum longitude of detection region
                    'lon2': 180, # maximum longitude
                    'lat1': -80, # minimum latitude
                    'lat2': -60, # maximum latitude
                    'res': 1., # resolution of the fields in km
                    'min_dep': 0, # minimum ocean depth where to look for eddies in m
                    'OW_thr': std, # 
                    'OW_thr_name': 'OW_std', # Okubo-Weiss threshold for eddy detection
                    'OW_thr_factor': -0.1, # Okubo-Weiss parameter threshold
                    'Npix_min': 100, # minimum number of pixels (grid cells) to be considered as eddy
                    'Npix_max': 9e10, # maximum number of pixels (grid cells)
                    'no_long': False, # If True, elongated shapes will not be considered
                    'no_two': False # If True, eddies with two minima in the OW
                                    # parameter and a OW > OW_thr in between  will not
                                    # be considered
                    }

def load_and_detect(year,day):
    
    d1=dt.datetime(year,1,1)
    dds=dt.timedelta(days=int(day))
    dde=dt.timedelta(days=int(day+1))
    ds=str(d1+dds)[:10]
    de=str(d1+dde)[:10]
    
    if day==365 and calendar.isleap(year) == False:
        return
    if os.path.isfile('PATH/TO/OUTPUT/so3_et_eddies_'+str(depth)+'_impfil_10-100km_'+ds+'_npm100_owthr0.1.pickle'):
        return
    
    uvdata=xr.open_dataset(datapath+'uv_uvgrid_'+str(depth)+'_10-100kmbp_'+str(year)+'_'+str(day).zfill(4)+'.nc')

    grid = xgcm.Grid(uvdata, 
                 coords={'X': {'center': 'XC', 'left': 'XG'},
                         'Y': {'center': 'YC', 'left': 'YG'},
                         'Z': {'center': 'Z' , 'right': 'Zp1'},
                         }, 
                 periodic={'X':True,'Y':False,'Z':False}, 
                 autoparse_metadata=False,metrics=metrics)
    data_OW=et.okuboweiss.calc(uvdata, grid, 'UVEL', 'VVEL')
    del uvdata

    mdata=data_OW.assign({'maskZ': (['Z','YG','XG'],oceanreg.reshape(1,*oceanreg.shape)),
                    'Depth': (['YG','XG'], bathyreg),
                    'OW':(['time','Z','YG','XG'],data_OW.OW.data),
                    'vort':(['time','Z','YG','XG'],data_OW.vort.data),
                    'OW_std': ([std])
                   })
    mdata=mdata.drop_vars(['UVEL','VVEL'])
    data_int=mdata.rename({'YG':'lat','XG':'lon'})
    del mdata

    depa=detection_parameters.copy()
    depa['start_time']=ds
    depa['end_time']=de

    eddies = et.detection.detect_OW(data_int.sel(Z=100, method='nearest'), depa, 'OW', 'vort', regrid_avoided=False, 
                                 use_bags=False, use_mp=False)[0]
    with open('/PATH/TO/OUTPUT/so3_et_eddies_'+str(depth)+'_impfil_10-100km_'+ds+'_npm100_owthr0.1.pickle','wb') as f:
        pickle.dump(eddies, f)


Parallel(n_jobs=155,batch_size=1,verbose=10)(delayed(load_and_detect)(year,day) for year in years for day in np.arange(366))


import numpy as np
import xarray as xr
import datetime as dt
import os
import sys
import math
import xgcm
import pyfesom2 as pf
import matplotlib.tri as mtri
from joblib import Parallel, delayed
import eddytools as et
import implicit_filter as imf

depth=19 #default depth index to interpolate
hlim=100 #upper limit for spatial filter in km
llim=10  #lower limit for spatial filter in km

#depth also set from batch script
year=int(sys.argv[1])
ind=int(sys.argv[2])
depth=int(sys.argv[3])

#helps to identify failed jobs
print('Year: '+str(year))
print('Year subset: '+str(ind))
print('Depth: '+str(depth))

# splitting the year into 31 day chunks works well. This can be changed for shorter/longer jobs.
inds=np.arange(ind*31,(ind*31)+31,1)

#deal with leap years
if inds[-1]>365:
    if year %4 ==0:
        inds=inds[inds<=365]
    else:
        inds=inds[inds<=364]

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
        print('grid type not supported')
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

#interpolation data (get once)
meshpath='/PATH/TO/MESH/'
mesh=pf.load_mesh(meshpath)
model_lons=mesh.x2
model_lats=mesh.y2
elements=mesh.elem.astype('int32')
d = model_lons[elements].max(axis=1) - model_lons[elements].min(axis=1)
no_cyclic_elem = np.argwhere(d < 100).ravel()

triang = mtri.Triangulation(model_lons, model_lats, elements[no_cyclic_elem])
tri = triang.get_trifinder()

diag=xr.open_dataset(meshpath+'fesom.mesh.diag.nc')
Zp1=diag.nz[depth].values
Z=diag.nz1[depth].values
del diag

#if Fesom crashed and recorded the day at restart again, we remove the duplicate
def remove_doubles(data,ly=False):
    if ly==True:
        data=data.sel(time=~((data.time.dt.month == 2) & (data.time.dt.day == 29)))
    dups=[]
    for i in np.arange(len(data.time)-1):
        if str(data.time[i].values)[:10]==str(data.time[i+1].values)[:10]:
            dups.append(i)
    if len(dups) > 0:
        newdata=data.drop_isel(time=dups)
        return newdata
    else:
        return data

dpath='/PATH/TO/DATA/'
# load u and v data. data is on nodes in this case.
def get_data(years,depths,removely=False,):
    
#     if os.path.isfile(datapath+'unod.fesom.'+str(years[0])+'.nc')==False:
#         print(datapath+'unod.fesom.'+str(years[0])+'.nc')
#         print('Data not found. Is your data stored on nodes?')
#         pass
    
    ufiles=[]
    vfiles=[]
    #for now only one year
    for year in years:
        ufiles.append(dpath+'unod.fesom.'+str(year)+'.nc')
        vfiles.append(dpath+'vnod.fesom.'+str(year)+'.nc')
        
    udata=xr.open_mfdataset(ufiles)
    vdata=xr.open_mfdataset(vfiles)
        

    
    return udata.unod[:,depth,:], vdata.vnod[:,depth,:]

udata,vdata=get_data([year],depth)    

# implicit filter    

cyclic = 1  # 1 if mesh is cyclic
cyclic_length = 360  # in degrees; if not cyclic, take it larger than  zonal size
cyclic_length = cyclic_length * math.pi / 180  # DO NOT TOUCH
meshtype = 'r'  # coordinates can be in physical measure 'm' or in radians 'r'
cartesian = False
r_earth = 6400

#lengthscales we filter. factor of 3.5 to convert to distance in km
smallscale= 2 * math.pi / (llim*3.5)
largescale = 2 * math.pi / (hlim*3.5)

jf = imf.JaxFilter()
jf.prepare(mesh.n2d, mesh.e2d, mesh.elem, mesh.x2, mesh.y2, meshtype, cartesian, cyclic_length, True)

def convert_to_reg_and_save(day):
    if os.path.isfile('/PATH/TO/OUTPUT/uv_uvgrid_'+str(Zp1)+'m_'+str(llim)+'-'+str(hlim)+'kmbp_'+str(year)+'_'+str(day).zfill(4)+'.nc') and os.path.getsize('/PATH/TO/OUTPUT/uv_uvgrid_'+str(Zp1)+'m_'+str(llim)+'-'+str(hlim)+'kmbp_'+str(year)+'_'+str(day).zfill(4)+'.nc')>18e8:
        print(str(year)+' file exists')
    else:
        
        #on elements for filter
        udelem=np.mean(udata.values[day,elements],axis=1)
        vdelem=np.mean(vdata.values[day,elements],axis=1)
        #filter and back to nodes
        ufill, vfill = jf.compute_velocity(1, smallscale, udelem, vdelem)
        ufilh, vfilh = jf.compute_velocity(1, largescale, udelem, vdelem)
        del udelem
        del vdelem

        uband=ufill-ufilh
        vband=vfill-vfilh

        del vfilh
        del ufilh
        del vfill
        del ufill

        #on uv points in this case. 
        uregoutu = np.asarray(mtri.LinearTriInterpolator(triang, uband,trifinder=tri)(upointslonl, upointslatl))
        vregoutv = np.asarray(mtri.LinearTriInterpolator(triang, vband,trifinder=tri)(vpointslonl, vpointslatl))

        dsuv = xr.Dataset(
        coords={'XC': (['XC'], lons_c),
                'XG': (['XG'], lons_gl),
                'YC': (['YC'], lats_c),
                'YG': (['YG'], lats_gl),
                'Z':  (['Z'], [-Z]),
                'Zp1': (['Zp1'], [-Zp1]),
                'time':(['time'],[udata.time[day].values])},
        data_vars={
                   'UVEL':  (['time','Z','YC','XG'], uregoutu.reshape(1,1,uregoutu.shape[0],uregoutu.shape[1])),
                   'VVEL':  (['time','Z','YG','XC'], vregoutv.reshape(1,1,vregoutv.shape[0],vregoutv.shape[1])),
                   'dxV':   (['YG','XG'], dxV),
                   'dyU':   (['YG','XG'], dyU),
                   'dxC':   (['YC','XG'], dxC),
                   'dyC':   (['YG','XC'], dyC),
                   'dxF':   (['YC','XC'], dxF),
                   'dyF':   (['YC','XC'], dyF),
                   'dxG':   (['YG','XC'], dxG),
                   'dyG':   (['YC','XG'], dyG),
                  }
        )
        dsuv.to_netcdf('/PATH/TO/OUTPUT/uv_uvgrid_'+str(Zp1)+'m_'+str(llim)+'-'+str(hlim)+'kmbp_'+str(year)+'_'+str(day).zfill(4)+'.nc')
        del uregoutu
        del vregoutv
        del dsuv

#the triangulation object prevents using other backends, solving this could speed things up significantly
Parallel(n_jobs=31,batch_size=1,verbose=10,backend='threading')(delayed(convert_to_reg_and_save)(day) for day in inds)


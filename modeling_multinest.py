
import numpy as np
from KinMS import KinMS
import uvutil,sample_vis
from astropy.io import fits
from scipy.optimize import minimize
import pymultinest
from galario.double import sampleImage


xs=3.#7.68
ys=3.#7.68
vs=2075
cellsize=.1
dv=83.
beamsize=.15

#inc=60.
#gassigma=30.
#posang=0.
#intflux=3.

sbrad=np.arange(0,2,.01)#sbrad=np.arange(0,.7,.15)

#bright_std=.8
#vmax=500
#vel_scale=.3

#sbprof=np.exp(-sbrad**2/2/(bright_std/2.355)**2)
velrad=sbrad
#velprof=vmax*np.arctan(velrad/vel_scale)/np.pi*2

diskthick=0
cleanout=True
nsamps=1e5
#setting up the stage for fitting
UVFITSfile='../CRLE_spw01_binned_comb_centered.uvfits'

vis_complex_data, wgt_data = uvutil.visload(UVFITSfile)
vis_complex_model_template=np.copy(vis_complex_data)
wgt_data=wgt_data.flatten()
good_vis=wgt_data>0
wgt_data=wgt_data[good_vis]
vis_complex_data=vis_complex_data.flatten()[good_vis]
modelheader = fits.getheader('../CRLE_spw01_binned_comb_centered.fits')
uu_cube, vv_cube, ww_cube = uvutil.uvload(UVFITSfile)
#this can be the same throughout
pcd = uvutil.pcdload(UVFITSfile)
y,x,zz=np.mgrid[:30,:30,:25]   #fixed this
#r=np.hypot(x-15+.326,y-15-.34)

'''

positive
posang 0:   to the right
posang 90: downwards
posang 180: to the left
posang 270 (=-90): upward

negative
posang 0:   to the left
posang 90: upward
posang 180: to the right
posang 270 (=-90): downward
'''


def loglike(cube,ndim,nparams):
   gassigma,bright_std,vmax,vel_scale,vel_cen,inc,posang,x_cen,y_cen,intflux=cube[0],cube[1],cube[2],cube[3],cube[4],cube[5],cube[6],cube[7],cube[8],cube[9]
   if True or 10<gassigma<700  and 0.1<bright_std<2 and 10<vmax<2000 and 0.1<vel_scale<2 and -300<vel_cen<300 and 0<inc<90   and 60<posang<120  and 1.<contF<20. and 0.1<contS<0.3 and -.15<x_cen<.15 and -.15<y_cen<.15:
	xcen=x+0.52351473-15
	ycen=y-0.33279625-15
	cc=np.cos(77.37057442/180*np.pi)
	ss=np.sin(77.37057442/180*np.pi)
	r2=(xcen*cc+ycen*ss)**2/(0.18297968/cellsize)**2+(xcen*ss-ycen*cc)**2/(0.11668485/cellsize)**2
	model_cont=5.37587948*1e-3*np.exp(-r2/2)/(2*np.pi*(0.18297968/cellsize)*(0.11668485/cellsize))
	sbprof=np.exp(-sbrad**2/2/(bright_std/2.355)**2)
	velprof=vmax*np.arctan(velrad/vel_scale)/np.pi*2
	#model_cont=contF*1e-3*np.exp(-r**2/2/(contS/cellsize)**2)/(2*np.pi*(contS/cellsize)**2)
	model=model_cont+KinMS(xs,ys,vs,cellSize=cellsize,dv=dv,beamSize=beamsize,inc=inc,gasSigma=gassigma,sbProf=sbprof,sbRad=sbrad,velRad=velrad,velProf=velprof,fileName=False,diskThick=diskthick,cleanOut=cleanout,ra=0,dec=0,nSamps=nsamps,posAng=posang,intFlux=intflux,inClouds=[],vLOS_clouds=[],flux_clouds=0,vSys=0,restFreq=115.271e9,phaseCen=np.array([x_cen,y_cen]),vOffset=vel_cen,fixSeed=False,vRadial=0,vPosAng=0,vPhaseCen=np.array([x_cen,y_cen]),returnClouds=False,gasGrav=False)
	xpos,ypos=256,256	
	model_padded=np.transpose(np.pad(model,((ypos-15,512-ypos-15),(xpos-15,512-xpos-15),(0,0)),mode='constant'),(2,0,1))
	#img=pyfits.open('my_img_mod.fits',mode='update')
	#img[0].data=model_padded
	#img.close()
	modelimage_cube = model_padded #fits.getdata(sbmodelloc)
	vis_complex_model=np.copy(vis_complex_model_template)
	for chan in range(modelimage_cube.shape[0]):
		uu=np.ones((uu_cube.shape[0],uu_cube.shape[1],1,uu_cube.shape[3]))
		vv=np.ones((vv_cube.shape[0],vv_cube.shape[1],1,vv_cube.shape[3]))
		uu[:,:,0,:]=uu_cube[:,:,chan,:]
		vv[:,:,0,:]=vv_cube[:,:,chan,:]
		modelimage=modelimage_cube[chan]
		uushape = uu.shape
		uu = uu.flatten()
		vv = vv.flatten()
		uu=uu.copy(order='C')  #This
		vv=vv.copy(order='C') #this
		modelimage=np.roll(np.flip(modelimage,axis=0),1,axis=0).copy(order='C')#.byteswap().newbyteorder()    #This
		model_complex = sampleImage(modelimage, np.absolute(modelheader['CDELT1'])/180*np.pi, uu, vv)  #this uses galario
		#model_complex = sample_vis.uvmodel(modelimage, modelheader, uu, vv, pcd)
		vis_complex = model_complex.reshape(uushape)
		vis_complex_model[:,:,chan,:]=vis_complex[:,:,0,:]
	#replace_visibilities('HZ10_spw01_comb.uvfits','my_img_mod.fits','model_visib.uvfits')
	#vis_complex_model,bb  = uvutil.visload('model_visib.uvfits')
	vis_complex_model=vis_complex_model.flatten()[good_vis]
	def find_param(scale):
		diff_all=np.abs(vis_complex_data-vis_complex_model*scale)
		return np.sum(wgt_data*diff_all*diff_all)
	#scale_analyt=np.sum(wgt_data*np.real(np.conjugate(vis_complex_data)*vis_complex_model))/np.sum(wgt_data*np.abs(vis_complex_model)**2)
	#chi2_all=diff_all*diff_all*wgt_data
	ln_like=-0.5*find_param(1)#scale_analyt)#result['fun']
	print ln_like
	return ln_like+2.4e6      #,scale[0]
   else:
	return -np.inf




n_params = 10


'''
a = pymultinest.Analyzer(outputfiles_basename= 'better_prior_model', n_params = 10)

samples=a.get_equal_weighted_posterior()[:,:-1]

a_lnZ = a.get_stats()['global evidence']
print 
print '************************'
print 'MAIN RESULT: Evidence Z '
print '************************'
print '  log Z for model with 1 line = %.1f' % (a_lnZ / log(10))
'''
#########################################################################################
#REDO with better priors

from Priors import Priors
pri=Priors()


#Example of priors
def prior(cube,ndim,nparams):
	cube[0]=pri.LogPrior(cube[0],10.,700.) 
	cube[1]=pri.LogPrior(cube[1],.1,2.)   
	cube[2]=pri.LogPrior(cube[2],10.,2000.)  
	cube[3]=pri.LogPrior(cube[3],.01,.3)   
	cube[4]=pri.UniformPrior(cube[4],-400.,400.)     
	cube[5]=pri.SinPrior(cube[5],0.,90.)    
	cube[6]=pri.UniformPrior(cube[6],60.,120.)      
	cube[7]=pri.UniformPrior(cube[7],-.15,.15)     
	cube[8]=pri.UniformPrior(cube[8],-.15,.15)   
	cube[9]=pri.LogPrior(cube[9],1,5) 




from galario import double
double.threads(6)


#need to change to "model" to get reliable evidence 
pymultinest.run(loglike, prior, n_params, outputfiles_basename='better_prior_model', resume = True, verbose = True,sampling_efficiency='model')

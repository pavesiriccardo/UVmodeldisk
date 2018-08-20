
import numpy as np
from KinMS import KinMS
import uvutil,sample_vis
from astropy.io import fits
from scipy.optimize import minimize
import pymultinest
from galario.double import sampleImage


xs=3.#7.68
ys=3.#7.68
vs=1660
cellsize=.1
dv=83.
beamsize=.15

#inc=60.
#gassigma=30.
#posang=0.
#intflux=4.5

sbrad=np.arange(0,2,.01)

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
UVFITSfile='HZ10_line_plus_cont.uvfits'

vis_complex_data, wgt_data = uvutil.visload(UVFITSfile)
vis_complex_model_template=np.copy(vis_complex_data)
wgt_data=wgt_data.flatten()
good_vis=wgt_data>0
wgt_data=wgt_data[good_vis]
vis_complex_data=vis_complex_data.flatten()[good_vis]
modelheader = fits.getheader('HZ10_line_plus_cont.fits')
uu_cube, vv_cube, ww_cube = uvutil.uvload(UVFITSfile)
#this can be the same throughout
pcd = uvutil.pcdload(UVFITSfile)
y,x,zz=np.mgrid[:30,:30,:50]
#cont1 at (256,255)
#cont2 at (263.5,254.5)
#r1=np.hypot(x-15+2.,y-15)
#r2=np.hypot(x-15-5.5,y-15+.5)

rCRLE=np.hypot(x-15+.326,y-15-.34)
model_contCRLE=6.48692467*1e-3*np.exp(-rCRLE**2/2/(0.16200506/cellsize)**2)/(2*np.pi*(0.16200506/cellsize)**2)
xpos,ypos=275,127	
model_contCRLE_pad=np.transpose(np.pad(model_contCRLE,((ypos-15,512-ypos-15),(xpos-15,512-xpos-15),(0,0)),mode='constant'),(2,0,1))	


def loglike(cube,ndim,nparams):
   gassigma,bright_std,vmax,vel_scale,vel_cen,inc,posang,intflux,x_cen,y_cen,x1,y1,x2,y2,contS1,contF1,contS2,contF2=cube[0],cube[1],cube[2],cube[3],cube[4],cube[5],cube[6],cube[7],cube[8],cube[9],cube[10],cube[11],cube[12],cube[13],cube[14],cube[15],cube[16],cube[17]
   print gassigma,bright_std,vmax,vel_scale,vel_cen,inc,posang,intflux,x_cen,y_cen,x1,y1,x2,y2,contS1,contF1,contS2,contF2
   if True or 10<gassigma<700  and 0.1<bright_std<2 and 10<vmax<2000 and 0.1<vel_scale<2 and -300<vel_cen<300 and 0<inc<90   and 60<posang<120  and 1.<contF<20. and 0.1<contS<0.3 and -.15<x_cen<.15 and -.15<y_cen<.15:
	#velprof=np.array([0]+list(velprof))
	sbprof=np.exp(-sbrad**2/2/(bright_std/2.355)**2)
	velprof=-vmax*np.arctan(velrad/vel_scale)/np.pi*2
	#model_cont=contF*1e-3*np.exp(-r**2/2/(contS/cellsize)**2)/(2*np.pi*(contS/cellsize)**2)
	model=KinMS(xs,ys,vs,cellSize=cellsize,dv=dv,beamSize=beamsize,inc=inc,gasSigma=gassigma,sbProf=sbprof,sbRad=sbrad,velRad=velrad,velProf=velprof,fileName=False,diskThick=diskthick,cleanOut=cleanout,ra=0,dec=0,nSamps=nsamps,posAng=posang,intFlux=intflux,inClouds=[],vLOS_clouds=[],flux_clouds=0,vSys=0,restFreq=115.271e9,phaseCen=np.array([x_cen,y_cen]),vOffset=vel_cen,fixSeed=False,vRadial=0,vPosAng=0,vPhaseCen=np.array([x_cen,y_cen]),returnClouds=False,gasGrav=False)
	r1=np.hypot(x-15-x1,y-15-y1)	
	r2=np.hypot(x-15-x2,y-15-y2)
	model_cont=contF1*1e-3*np.exp(-r1**2/2/(contS1/cellsize)**2)/(2*np.pi*(contS1/cellsize)**2)
	model_cont+=contF2*1e-3*np.exp(-r2**2/2/(contS2/cellsize)**2)/(2*np.pi*(contS2/cellsize)**2)
	model_cont[:,:,13:33]+=model
	xpos,ypos=258,255
	model_padded=np.transpose(np.pad(model_cont,((ypos-15,512-ypos-15),(xpos-15,512-xpos-15),(0,0)),mode='constant'),(2,0,1))+model_contCRLE_pad
	#img=pyfits.open('my_img_mod.fits',mode='update')
	#img[0].data=model_padded
	#img.close()
	modelimage_cube = model_padded #fits.getdata(sbmodelloc)
	vis_complex_model=np.copy(vis_complex_model_template)
	for chan in range(modelimage_cube.shape[0]):
	    if True: #chan<12 or chan>32:
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
		model_complex = sampleImage(modelimage, np.absolute(modelheader['CDELT1'])/180*np.pi, uu, vv) 
		#model_complex = sample_vis.uvmodel(modelimage, modelheader, uu, vv, pcd)
		vis_complex = model_complex.reshape(uushape)
		vis_complex_model[:,:,chan,:]=vis_complex[:,:,0,:]
	    else:
		vis_complex_model[:,:,chan,:]=vis_complex_model_template[:,:,chan,:]
	#replace_visibilities('HZ10_spw01_comb.uvfits','my_img_mod.fits','model_visib.uvfits')
	#vis_complex_model,bb  = uvutil.visload('model_visib.uvfits')
	vis_complex_model=vis_complex_model.flatten()[good_vis]
	def find_param(scale):
		diff_all=np.abs(vis_complex_data-vis_complex_model*scale)
		return np.sum(wgt_data*diff_all*diff_all)
	#from scipy.optimize import minimize
	#result=minimize(find_param,1.)
	#scale=result['x']
	#scale_analyt=np.sum(wgt_data*np.real(np.conjugate(vis_complex_data)*vis_complex_model))/np.sum(wgt_data*np.abs(vis_complex_model)**2)
	#print scale_analyt
	#chi2_all=diff_all*diff_all*wgt_data
	ln_like=-0.5*find_param(1)#result['fun']
	print ln_like+2.5e6
	return ln_like+2.5e6  #,scale[0]
   else:
	return -np.inf




n_params = 18

'''

a = pymultinest.Analyzer(outputfiles_basename='mytest' + 'better_prior_model_withCRLE_fixed', n_params = 18)

samples=a.get_equal_weighted_posterior()[:,:-1]


import corner
fig = corner.corner(samples, labels=["gas $\sigma$","bright std.","vmax","vscale","v0","inc.","pos_ang","cont1","cont2","x_pos","y_pos"])


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

#sample priors
def prior(cube,ndim,nparams):
	cube[0]=pri.LogPrior(cube[0],10.,700.) 
	cube[1]=pri.LogPrior(cube[1],.1,2.)   
	cube[2]=pri.LogPrior(cube[2],10.,2000.)  
	cube[3]=pri.LogPrior(cube[3],.1,2.)   
	cube[4]=pri.UniformPrior(cube[4],-300.,300.)      
	cube[5]=pri.SinPrior(cube[5],0.,90.)   
	cube[6]=pri.UniformPrior(cube[6],-30.,30.)      
	cube[7]=pri.LogPrior(cube[7],1,10)    
	cube[8]=pri.UniformPrior(cube[8],-.2,.2)       
	cube[9]=pri.UniformPrior(cube[9],-.2,.2) 
	#x1,y1,x2,y2,contS1,contF1,contS2,contF2      #these are 1 and 99 percentiles of continuum only modeling
	cube[10]=pri.UniformPrior(cube[10],-1.78453893,-0.60732722) 
	cube[11]=pri.UniformPrior(cube[11],-0.34511807, 0.68385303)   
	cube[12]=pri.UniformPrior(cube[12],4.87697932,  5.9924711)  
	cube[13]=pri.UniformPrior(cube[13],-0.54919645, 0.2639608)   
	cube[14]=pri.UniformPrior(cube[14], 0.04479321,  0.18394208)      
	cube[15]=pri.UniformPrior(cube[15],0.48801719, 0.8252569 )    
	cube[16]=pri.UniformPrior(cube[16],0.02017528, 0.09588096)      
	cube[17]=pri.UniformPrior(cube[17],0.38141584,0.75691156)     





from galario import double
double.threads(2)

#need to change to "model" to get reliable evidence 
pymultinest.run(loglike, prior, n_params, outputfiles_basename='mytest' + 'better_prior_model_withCRLE_fixed', resume = True, verbose = True,sampling_efficiency='model')

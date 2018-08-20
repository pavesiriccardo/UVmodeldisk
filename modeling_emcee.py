
import numpy as np
from KinMS import KinMS
import uvutil,sample_vis
from astropy.io import fits
from scipy.optimize import minimize
import pymultinest


xs=3.#7.68
ys=3.#7.68
vs=1660
cellsize=.1
dv=83.
beamsize=.15


#intflux=4.5

sbrad=np.arange(0,2,.01)
velrad=sbrad


diskthick=0
cleanout=True
nsamps=1e5
#setting up the stage for fitting
UVFITSfile='HZ10_spw01_binned_comb.uvfits'

vis_complex_data, wgt_data = uvutil.visload(UVFITSfile)
vis_complex_model_template=np.copy(vis_complex_data)
wgt_data=wgt_data.flatten()
good_vis=wgt_data>0
wgt_data=wgt_data[good_vis]
vis_complex_data=vis_complex_data.flatten()[good_vis]
modelheader = fits.getheader('HZ10_spw01_binned_comb.fits')
uu_cube, vv_cube, ww_cube = uvutil.uvload(UVFITSfile)
#this can be the same throughout
pcd = uvutil.pcdload(UVFITSfile)
x,y,zz=np.mgrid[:30,:30,:20]
#cont1 at (256,255)
#cont2 at (263.5,254.5)
#r1=np.hypot(x-15+2.,y-15)
#r2=np.hypot(x-15-5.5,y-15+.5)

def point_source(temp,x_cen,y_cen,chan_cen,fwhm,flux):
	small_cube=np.copy(temp)
	gau1d=np.exp(-(np.arange(20)-chan_cen)**2/2/(fwhm/2.355)**2)
	gau1d*=flux#/np.sum(gau1d)
	xint=int(x_cen)
	yint=int(y_cen)
	xfrac=x_cen-xint
	yfrac=y_cen-yint
	small_cube[yint,xint,:]+=(1-xfrac)*(1-yfrac)*gau1d
	small_cube[yint+1,xint,:]+=(1-xfrac)*(yfrac)*gau1d
	small_cube[yint,xint+1,:]+=(xfrac)*(1-yfrac)*gau1d
	small_cube[yint+1,1+xint,:]+=(xfrac)*(yfrac)*gau1d
	return small_cube



def loglike(cube):
  gassigma1,bright_std1,vmax1,vel_scale1,inc1,posang1,intflux1,gassigma2,bright_std2,vmax2,vel_scale2,inc2,posang2=cube[0],cube[1],cube[2],cube[3],cube[4],cube[5],cube[6],cube[7],cube[8],cube[9],cube[10],cube[11],cube[12]
  print gassigma1,bright_std1,vmax1,vel_scale1,inc1,posang1,intflux1,gassigma2,bright_std2,vmax2,vel_scale2,inc2,posang2
  if 5<gassigma1<200  and 0.02<bright_std1<2 and 1<vmax1<800 and 0.005<vel_scale1<.4 and 0<inc1<90   and -180<posang1<180  and .3<intflux1<6.5 and 5<gassigma2<200  and 0.02<bright_std2<2 and 1<vmax2<800 and 0.005<vel_scale2<.4 and 0<inc2<90   and -180<posang2<180 :	
	#chan_cen1,chan_cen2,fwhm1,fwhm2,flux1,flux2,contF1,contF2=cube[0],cube[1],cube[2],cube[3],cube[4],cube[5],cube[6],cube[7]
	sbprof1=np.exp(-sbrad**2/2/(bright_std1/2.355)**2)
	sbprof2=np.exp(-sbrad**2/2/(bright_std2/2.355)**2)
	velprof1=-vmax1*np.arctan(velrad/vel_scale1)/np.pi*2
	velprof2=-vmax2*np.arctan(velrad/vel_scale2)/np.pi*2
	#model_cont=contF*1e-3*np.exp(-r**2/2/(contS/cellsize)**2)/(2*np.pi*(contS/cellsize)**2)
	x_cen,y_cen,vel_cen=-0.25*cellsize,-1.27*cellsize,(10.4-10)*dv
	cube1=KinMS(xs,ys,vs,cellSize=cellsize,dv=dv,beamSize=beamsize,inc=inc1,gasSigma=gassigma1,sbProf=sbprof1,sbRad=sbrad,velRad=velrad,velProf=velprof1,fileName=False,diskThick=diskthick,cleanOut=cleanout,ra=0,dec=0,nSamps=nsamps,posAng=posang1,intFlux=intflux1,inClouds=[],vLOS_clouds=[],flux_clouds=0,vSys=0,restFreq=115.271e9,phaseCen=np.array([x_cen,y_cen]),vOffset=vel_cen,fixSeed=False,vRadial=0,vPosAng=0,vPhaseCen=np.array([x_cen,y_cen]),returnClouds=False,gasGrav=False)
	x_cen,y_cen,vel_cen=-0.47*cellsize,5.05*cellsize,(6.7-10)*dv
	cube2=KinMS(xs,ys,vs,cellSize=cellsize,dv=dv,beamSize=beamsize,inc=inc2,gasSigma=gassigma2,sbProf=sbprof2,sbRad=sbrad,velRad=velrad,velProf=velprof2,fileName=False,diskThick=diskthick,cleanOut=cleanout,ra=0,dec=0,nSamps=nsamps,posAng=posang2,intFlux=(4.5-intflux1),inClouds=[],vLOS_clouds=[],flux_clouds=0,vSys=0,restFreq=115.271e9,phaseCen=np.array([x_cen,y_cen]),vOffset=vel_cen,fixSeed=False,vRadial=0,vPosAng=0,vPhaseCen=np.array([x_cen,y_cen]),returnClouds=False,gasGrav=False)	
	#model2=point_source(model,15-1.27,15-.25,chan_cen1,fwhm1,flux1*1e-3)
	#model2=point_source(model2,15+5.05,15-.47,chan_cen2,fwhm2,flux2*1e-3)
	model2=point_source(cube1+cube2,15-1.27,15-.25,10,1000,.55*1e-3)	
	model2=point_source(model2,15+5.05,15-.47,10,1000,.42*1e-3)	
	xpos,ypos=258,255	
	model_padded=np.transpose(np.pad(model2,((ypos-15,512-ypos-15),(xpos-15,512-xpos-15),(0,0)),mode='constant'),(2,0,1))
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
		model_complex = sample_vis.uvmodel(modelimage, modelheader, uu, vv, pcd)
		vis_complex = model_complex.reshape(uushape)
		vis_complex_model[:,:,chan,:]=vis_complex[:,:,0,:]
	#replace_visibilities('HZ10_spw01_comb.uvfits','my_img_mod.fits','model_visib.uvfits')
	#vis_complex_model,bb  = uvutil.visload('model_visib.uvfits')
	vis_complex_model=vis_complex_model.flatten()[good_vis]
	def find_param(scale):
		diff_all=np.abs(vis_complex_data-vis_complex_model*scale)
		return np.sum(wgt_data*diff_all*diff_all)
	#from scipy.optimize import minimize
	#result=minimize(find_param,1.)
	#scale=result['x']
	scale_analyt=np.sum(wgt_data*np.real(np.conjugate(vis_complex_data)*vis_complex_model))/np.sum(wgt_data*np.abs(vis_complex_model)**2)
	#chi2_all=diff_all*diff_all*wgt_data
	ln_like=-0.5*find_param(scale_analyt)#result['fun']
	print scale_analyt,ln_like
	return ln_like#,scale[0]
  else:
	return -np.inf



n_params = 13

# run MultiNest
#pymultinest.run(loglike, prior, n_params, outputfiles_basename='mytest' + '_1_', resume = False, verbose = True,sampling_efficiency='parameter')


'''
a = pymultinest.Analyzer(outputfiles_basename='mytest' + 'better_prior_model', n_params = n_params)

samples=a.get_equal_weighted_posterior()[:,:-1]


import corner
fig = corner.corner(samples, labels=["chan_cen1","chan_cen2","fwhm1","fwhm2","flux1","flux2","contF1","contF2"])


a_lnZ = a.get_stats()['global evidence']
print 
print '************************'
print 'MAIN RESULT: Evidence Z '
print '************************'
print '  log Z for model with 1 line = %.1f' % (a_lnZ / log(10))
'''
#########################################################################################
#REDO with better priors



from Priors_emcee import Priors
pri=Priors()

def prior(cube):
	lnprior=0
	lnprior+=pri.LogPrior(cube[0],10.,100.) 
	lnprior+=pri.LogPrior(cube[1],.05,1.)   
	lnprior+=pri.LogPrior(cube[2],10.,500.) 
	lnprior+=pri.LogPrior(cube[3],.01,.2)   
	lnprior+=pri.SinPrior(cube[4],0.,90.)   
	lnprior+=pri.UniformPrior(cube[5],-180.,180.)    
	lnprior+=pri.UniformPrior(cube[6],1.,4.5)
	lnprior+=pri.LogPrior(cube[7],10.,100.) 
	lnprior+=pri.LogPrior(cube[8],.05,1.)   
	lnprior+=pri.LogPrior(cube[9],10.,500.) 
	lnprior+=pri.LogPrior(cube[10],.01,.2)   
	lnprior+=pri.SinPrior(cube[11],0.,90.)   
	lnprior+=pri.UniformPrior(cube[12],-180.,180.)
	return lnprior

	


def lnprob(cube):
	if prior(cube)>-np.inf: 
		return prior(cube)+loglike(cube)
	else:
		return -np.inf




nll = lambda *args: -lnprob(*args)
#result=minimize(nll,np.array([60,30.,0,4.5,.8,500,.3]),method='Nelder-Mead')   #this works, even with 7 parameters!


ndim, nwalkers = 13, 30
#pos = [result['x'] *(1+.1*np.random.randn(ndim)) for i in range(nwalkers)] #result["x"]
pos = [result*(1+.1*np.random.randn(ndim)) for i in range(nwalkers)] #result["x"]


import emcee
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,threads=30)
for idx in range(20):
	sampler.run_mcmc(pos, 100)
	samples = sampler.chain[:, 10:, :].reshape((-1, ndim))
	np.save('samples_max',samples)
	print idx



import corner
fig = corner.corner(samples, labels=["$\sigma$1","bright std.1","vmax1","vscale1","inc.1","pos_ang1","intflux1","$\sigma$2","bright std.2","vmax2","vscale2","inc.2","pos_ang2"])







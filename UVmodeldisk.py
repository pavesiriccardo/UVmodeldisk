"""
Copyright (C) 2018, Riccardo Pavesi
E-mail: pavesiriccardo3 -at- gmail.com
Updated versions of the software are available through github:
https://github.com/pavesiriccardo/uvmodeldisk
 
If you have found this software useful for your research,
I would appreciate an acknowledgment to the use of the
"Disk modeling in the UV plane (UVmodeldisk) routines of Pavesi et al., (2018a)".
[https://arxiv.org/abs/1803.08048]
This software is provided as is without any warranty whatsoever.
For details of permissions granted please see LICENCE.md
"""

import numpy as np
from KinMS import KinMS
import uvutil
from astropy.io import fits
import pymultinest,emcee
from galario.double import sampleImage

class uvmodeldisk(object):
	def __init__(self,xs,ys,vs,cellsize,dv,UVFITSfile,cube_template_file):
		"""
		This function initializes the modeling object.
		Parameters
		----------
		xs: float
			X-axis size of the model image (in arcsec)
		ys: float
			Y-axis size of the model image (in arcsec)
		vs: float
			Velocity-axis size of the model image (in km/s)
		cellsize: float
			Arcseconds in a pixel
		dv: float
			Km/s in a channel
		UVFITSfile: string
			Filename of the UVFITS visibility file
		cube_template_file: string
			Filename of the image cube to be used as a model template (the dimensions and coordinates are the only thing that's used).
			It's best obtained by imaging the visibilities themselves, using mode='channel'. This ensures that the cell size is sufficient to suitable for the UV coverage.
		Returns
		-------
			    
		"""
		self.xs=xs   
		self.ys=ys   #y-axis size of the model image (in arcsec)
		self.vs=vs
		self.cellsize=cellsize
		self.dv=dv
		self.UVFITSfile=UVFITSfile #'.uvfits'
		self.cube_template_file=cube_template_file #'.fits'
		self.sbrad=np.arange(0,2,.01)   #radius vector defining the light profile, sbprof is the light intensity evaluated at these radial values
		self.velrad=self.sbrad          #use the same radius vector for the velocity definition, used by velprof to define the velocity profile
		self.nsamps=1e5                 #number of clouds, should be sufficient
		self.vis_complex_data, self.wgt_data = uvutil.visload(self.UVFITSfile) #load visibility and weights
		self.vis_complex_model_template=np.copy(self.vis_complex_data)
		self.wgt_data=self.wgt_data.flatten()
		self.good_vis=self.wgt_data>0
		self.wgt_data=self.wgt_data[self.good_vis]  #get rid of flagged data
		self.vis_complex_data=self.vis_complex_data.flatten()[self.good_vis]
		self.modelheader = fits.getheader(self.cube_template_file)   
		self.uu_cube, self.vv_cube, self.ww_cube = uvutil.uvload(self.UVFITSfile)
		self.pcd = uvutil.pcdload(self.UVFITSfile)
		self.Nxpix=self.modelheader['NAXIS1']
		self.Nypix=self.modelheader['NAXIS2']
		self.Nchan=self.modelheader['NAXIS3']
		self.Nxpix_small=int(self.xs/self.cellsize)
		self.Nypix_small=int(self.ys/self.cellsize)
		self.Nchan_small=int(self.vs/self.dv)
		#In this code we assume the continuum emission can be modelled as a 2D Gaussian, with known parameters. This is often the case and the continuum parameters are often precisely known. If they are not, include the continuum parameters as fitting parameters and make sure the visibilities cover a good range of continuum-only channels. 
		self.y,self.x,self.zz=np.mgrid[:self.Nypix_small,:self.Nxpix_small,:self.Nchan_small]   #define a coordinate cube of the same size as the KinMS model, to make the continuum model, to be added in.
		self.offset_lnlike=0
		self.xpos_center_padded,self.ypos_center_padded= int(self.Nxpix/2),int(self.Nypix/2)
	def set_ellipt_gauss_continuum(self,cont_xcen,cont_ycen,cont_maj_FWHM,cont_min_FWHM,cont_posang,cont_flux):
		"""
		    This function defines the continuum model.
		    Parameters
		    ----------
		    cont_xcen,cont_ycen: (float,float)
		    	X-axis and Y-axis offset of continuum center from image center (in arcsec)
		    cont_maj_FWHM,cont_min_FWHM: (float,float)
		    	FWHM of the major and minor axes of the continuum 2D Gaussian (in arcsec)
		    cont_posang: float
		    	Position angle of the ellipse, counterclockwise from the positive x axis (in degrees)
		    cont_flux: float
		    	Integrated flux of the continuum emission (in mJy)
		    Returns
		    -------
		    
			    
		"""
		xcen=self.x-cont_xcen/self.cellsize-self.Nxpix_small/2.
		ycen=self.y-cont_ycen/self.cellsize-self.Nypix_small/2.
		cc=np.cos(cont_posang/180*np.pi)
		ss=np.sin(cont_posang/180*np.pi)
		r2=(xcen*cc+ycen*ss)**2/(cont_maj_FWHM/self.cellsize/2.355)**2+(xcen*ss-ycen*cc)**2/(cont_min_FWHM/self.cellsize/2.355)**2
		self.model_cont=cont_flux*1e-3*np.exp(-r2/2)/(2*np.pi*(cont_maj_FWHM/self.cellsize/2.355)*(cont_min_FWHM/self.cellsize/2.355))
	def loglike_multinest(self,cube,ndim,npar):
		return self.loglike(cube)
	def my_prior(self,cube):
		return 0
	def loglike(self,cube):

		'''
		This function calculates the model ln(likelihood).
		Parameters
		----------
		gassigma: float
			The gas dispersion in km/s
		bright_std: float
			The size as gaussian FWHM in arcsec
		vmax: float
			The max velocity in km/s, the asymptote of the arctan
		vel_scale: float
			The radial distance, in arcsec, where velocity goes to vmax/2
		vel_cen: float
			The velocity center in km/s away from central channel, increasing vel_cen moves it to later channels, i.e. same direction as the channel ordering
		inc: float
		 	Inclination, is 0 for face on and 90 for edge-on
		posang: float
			Position angle starting with red emission from horizontal to the right (toward decreasing RA), and increasing counterclockwise (when we have a - in front of vmax). posang near 0, and - in front of vmax, means the emission moves right to left as channels increase. positive posang rotates this pattern toward the north, in a counterclockwise manner
		x_cen, y_cen: (float,float)
			Center position for the disk, they are in arcsec. y_cen actually controls the x-axis and positive means increasing x of center. x_cen controls y axis, and positive means increasing y of center
		intflux: float
			Line integrated flux in Jy km/s
		Returns
		-------
			ln_like+self.offset_lnlike: float
				We offset the ln_like value by a constant to make the number be small enough for Multinest to be able to work with.
		Definition of posang for the rotating disk:
		If velprof=vmax*np.arctan(velrad/vel_scale)/np.pi*2
		posang 0:   to the right
		posang 90: downwards
		posang 180: to the left
		posang 270 (=-90): upward
		If velprof=-vmax*np.arctan(velrad/vel_scale)/np.pi*2
		posang 0:   to the left
		posang 90: upward
		posang 180: to the right
		posang 270 (=-90): downward
		For example: if the emission is moving from left to right, as channels increase (toward lower frequency and higher velocity, red). Then need posang=180 if minus sign in front of vmax.
		'''
		gassigma,bright_std,vmax,vel_scale,vel_cen,inc,posang,x_cen,y_cen,intflux=cube[0],cube[1],cube[2],cube[3],cube[4],cube[5],cube[6],cube[7],cube[8],cube[9]
		model_cont=self.model_cont
		sbprof=np.exp(-self.sbrad**2/2/(bright_std/2.355)**2)
		velprof=vmax*np.arctan(self.velrad/vel_scale)/np.pi*2
		model=model_cont+KinMS(self.xs,self.ys,self.vs,cellSize=self.cellsize,dv=self.dv,beamSize=0,inc=inc,gasSigma=gassigma,sbProf=sbprof,sbRad=self.sbrad,velRad=self.velrad,velProf=velprof,fileName=False,diskThick=0,cleanOut=True,ra=0,dec=0,nSamps=self.nsamps,posAng=posang,intFlux=intflux,inClouds=[],vLOS_clouds=[],flux_clouds=0,vSys=0,restFreq=115.271e9,phaseCen=np.array([x_cen,y_cen]),vOffset=vel_cen,fixSeed=False,vRadial=0,vPosAng=0,vPhaseCen=np.array([x_cen,y_cen]),returnClouds=False,gasGrav=False)
		xpos,ypos=self.xpos_center_padded,self.ypos_center_padded
		model_padded=np.transpose(np.pad(model,((ypos-self.Nypix_small/2,self.Nypix-ypos-self.Nypix_small/2),(xpos-self.Nxpix_small/2,self.Nxpix-xpos-self.Nxpix_small/2),(0,0)),mode='constant'),(2,0,1))
		modelimage_cube = model_padded 
		vis_complex_model=np.copy(self.vis_complex_model_template)
		for chan in range(modelimage_cube.shape[0]):
			uu=np.ones((self.uu_cube.shape[0],self.uu_cube.shape[1],1,self.uu_cube.shape[3]))
			vv=np.ones((self.vv_cube.shape[0],self.vv_cube.shape[1],1,self.vv_cube.shape[3]))
			uu[:,:,0,:]=self.uu_cube[:,:,chan,:]
			vv[:,:,0,:]=self.vv_cube[:,:,chan,:]
			modelimage=modelimage_cube[chan]
			uushape = uu.shape
			uu = uu.flatten()
			vv = vv.flatten()
			uu=uu.copy(order='C')  #This
			vv=vv.copy(order='C') #this
			modelimage=np.roll(np.flip(modelimage,axis=0),1,axis=0).copy(order='C')#.byteswap().newbyteorder()    #This
			model_complex = sampleImage(modelimage, np.absolute(self.modelheader['CDELT1'])/180*np.pi, uu, vv)  #this uses galario
			#model_complex = sample_vis.uvmodel(modelimage, modelheader, uu, vv, pcd)
			vis_complex = model_complex.reshape(uushape)
			vis_complex_model[:,:,chan,:]=vis_complex[:,:,0,:]
		#replace_visibilities('HZ10_spw01_comb.uvfits','my_img_mod.fits','model_visib.uvfits')
		#vis_complex_model,bb  = uvutil.visload('model_visib.uvfits')
		vis_complex_model=vis_complex_model.flatten()[self.good_vis]
		def find_param(scale):
			diff_all=np.abs(self.vis_complex_data-vis_complex_model*scale)
			return np.sum(self.wgt_data*diff_all*diff_all)
		ln_like=-0.5*find_param(1)
		print(ln_like)
		return ln_like+self.offset_lnlike
	def run_Multinest(self,Nthreads,output_filename='temporary_model'):
		'''
		This function runs Multinest using the self.my_prior priors, which need to be appropriately set to a Multinest-type of priors.
		
		Parameters
		Nthreads: int
			Number of threads per Multinest instance used by Galario to compute the model visibilities
		Output_filename: string
			Multinest output base file names. See pymultinest docs for how to read and analyze those.
		
		'''
		n_params = 10
		from galario import double
		double.threads(Nthreads)
		pymultinest.run(self.loglike_multinest, self.my_prior, n_params, outputfiles_basename=output_filename, resume = True, verbose = True,sampling_efficiency='model')
	def lnprob(self,cube):
		'''
		This function combines the emcee-type prior and the log likelihood to compute the log probability. Should be used with emcee.
		'''
		if self.my_prior(cube)>-np.inf: 
			return self.my_prior(cube)+self.loglike(cube)
		else:
			return -np.inf
	def find_max_prob(self,starting_guess,Nthreads_galario):
		'''
		This function tries to find the maximum probability parameter values. This may be useful for ML fitting or, e.g., to start an MCMC sampler such as emcee.
		Parameters
		
		Nthreads_galario: int
			Number of threads to be used by Galario
		starting_guess: list of float
			This should be a list of disk fitting parameters to be used as a starting point for the optimizer.
		
		Returns
		
		result['x']: list of float
			The list of parameters which achieves the highest probability, as computed by the optimizer.
		'''
		from galario import double
		double.threads(Nthreads_galario)
		nll = lambda *args: -self.lnprob(*args)
		from scipy.optimize import minimize
		result=minimize(nll,starting_guess,method='Nelder-Mead')
		print(result)
		return result['x']
	def run_emcee(self,Nthreads_galario,Nthreads_emcee,nwalkers,starting_point,Nsteps,output_filename='temporary_model'):
		'''
		This function runs emcee to produce MCMC samples from the probability distribution defined by my_prior and loglike.
		
		Parameters
		
		Nthreads_galario, Nthreads_emcee: (int,int)
			Number of threads to be used by Galario and by emcee, respectively
		nwalkers: int
			Number of Emcee walkers, see emcee documentation
		starting_point: list of float
			List of parameters to be used as starting point for the MCMC chain
		Nsteps: int
			Number of MCMC steps, needs to be >100 because I burn 100 by default (easy to change this below). The functions saves the samples every 100 steps.
		output_filename: string
			Filename for the numpy save file storing the samples
		'''
		ndim=10
		from galario import double
		double.threads(Nthreads_galario)
		pos = [np.array(starting_point)*(1+.1*np.random.randn(ndim)) for i in range(nwalkers)]
		sampler = emcee.EnsembleSampler(nwalkers, ndim, picklable_boundmethod(self.lnprob),threads=Nthreads_emcee)
		sampler.run_mcmc(pos, 100)
		for idx in range(int(Nsteps/100)):
			sampler.run_mcmc(None, 100)
			samples = sampler.chain[:, 10:, :].reshape((-1, ndim))
			np.save(output_filename,samples)
			print(idx,' of ',int(Nsteps/100))



#Necessary in order to use the lnprob instance method within the emcee sampler, to make it picklable
class picklable_boundmethod(object):
	def __init__(self, mt):
		self.mt = mt
	def __getstate__(self):
		return self.mt.im_self, self.mt.im_func.__name__
	def __setstate__(self, (s,fn)):
		self.mt = getattr(s, fn)
	def __call__(self, *a, **kw):
		return self.mt(*a, **kw)



'''
#Example of priors for Multinest
def prior(cube,ndim,nparams):
	from Priors_multinest import Priors
	pri=Priors()
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

#Example of priors for emcee
def prior(cube):
	from Priors_emcee import Priors
	pri=Priors()
	lnprior=0
	lnprior+=pri.LogPrior(cube[0],10.,700.) 
	lnprior+=pri.LogPrior(cube[1],.1,2.)   
	lnprior+=pri.LogPrior(cube[2],10.,2000.)  
	lnprior+=pri.LogPrior(cube[3],.01,.3)   
	lnprior+=pri.UniformPrior(cube[4],-400.,400.)      
	lnprior+=pri.SinPrior(cube[5],0.,90.)   
	lnprior+=pri.UniformPrior(cube[6],60.,120.)     
	lnprior+=pri.UniformPrior(cube[7],-.15,.15)     
	lnprior+=pri.UniformPrior(cube[8],-.15,.15)    
	lnprior+=pri.LogPrior(cube[9],1.,5.)       
	return lnprior
'''

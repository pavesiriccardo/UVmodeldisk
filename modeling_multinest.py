
import numpy as np
from KinMS import KinMS
import uvutil
from astropy.io import fits
import pymultinest
from galario.double import sampleImage


xs=3. #x-axis size of the model image (in arcsec)
ys=3. #y-axis size of the model image (in arcsec)
vs=2075 #velocity-axis size of the model image (in km/s)
cellsize=.1  #arcsec in a pixel
dv=83.   #km/s in a channel

sbrad=np.arange(0,2,.01) #radius vector defining the light profile, sbprof is the light intensity evaluated at these radial values

velrad=sbrad #use the same radius vector for the velocity definition, used by velprof to define the velocity profile

diskthick=0
cleanout=True  #no beam convolution required
nsamps=1e5   #number of clouds, should be sufficient

#setting up the stage for fitting
UVFITSfile='data/filename.uvfits'   #where the visibility data are

vis_complex_data, wgt_data = uvutil.visload(UVFITSfile) #load visibility and weights
vis_complex_model_template=np.copy(vis_complex_data)
wgt_data=wgt_data.flatten()
good_vis=wgt_data>0
wgt_data=wgt_data[good_vis]  #get rid of flagged data
vis_complex_data=vis_complex_data.flatten()[good_vis]
modelheader = fits.getheader('data/image_cube.fits')   #this FITS file should contain the image cube to be used as a model template (the dimensions and coordinates are the only thing that's used). It's best obtained by imaging the visibilities themselves, using mode='channel'. This ensures that the cell size is sufficient to suitable for the UV coverage.
uu_cube, vv_cube, ww_cube = uvutil.uvload(UVFITSfile)
pcd = uvutil.pcdload(UVFITSfile)

#In this code we assume the continuum emission can be modelled as a 2D Gaussian, with known parameters. This is often the case and the continuum parameters are often precisely known. If they are not, include the continuum parameters as fitting parameters and make sure the visibilities cover a good range of continuum-only channels. 
y,x,zz=np.mgrid[:int(ys/cellsize),:int(xs/cellsize),:int(vs/dv)]   #define a coordinate cube of the same size as the KinMS model, to make the continuum model, to be added in.

'''
Definition of posang for the rotating disk.

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
'''

#For example: if the emission is moving from left to right, as channels increase (toward lower frequency and higher velocity, red). Then need posang=180 if minus sign in front of vmax.

#gassigma is the gas dispersion in km/s
#bright_std is the size as gaussian FWHM in arcsec
#vmax is max velocity in km/s
#vel_scale is the radial distance, in arcsec, where velocity goes to vmax/2
#vel_cen is km/s away from central channel, increasing vel_cen moves it to later channels, i.e. same direction as the channel ordering
#inc is 0 for face on and 90 for edge-on
#posang is starting with red emission from horizontal to the right (toward decreasing RA), and increasing counterclockwise (when we have a - in front of vmax). posang near 0, and - in front of vmax, means the emission moves right to left as channels increase. positive posang rotates this pattern toward the north, in a counterclockwise manner
#contF is the continuum integrated flux in mJy
#contS is the continuum size as standard deviation in arcsec
#x_cen, y_cen are in arcsec. y_cen actually controls the x-axis and positive means increasing x of center. x_cen controls y axis, and positive means increasing y of center


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

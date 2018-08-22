import numpy as np
from KinMS import KinMS
import uvutil
from astropy.io import fits
import emcee
from galario.double import sampleImage


from UVmodeldisk import uvmodeldisk  #import the main class


my_model=uvmodeldisk(6.,6.,4000.,.2,100.,'../data/gn10-C.statwt_selec.uvfits','../data/images/gn10-C.fits')  #Defines the model parameters, see class constructor

my_model.xpos_center_padded=126 #edit approximate center pixel for centering the KinMS model
my_model.ypos_center_padded=128

my_model.set_ellipt_gauss_continuum(-.012,-.068,.471,.471,0,8.)   #set the continuum 2D elliptical Gaussian model

#my_model.loglike([200.,.1,400.,.01,0,60.,180,0,0,10.])   #check that the lok likelihood works

#Define appropriate priors for each parameter

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

my_model.my_prior=prior   #Set the my_prior function of the object to point to the correct prior definition


best_parameters=my_model.find_max_prob([50.,.5,300.,.05,50,60.,90.,0.01,0.01,3.],30)   #Look for the Maximum Probability solution


#Run the MCMC with 2 threads per Galario instance, 16 threads for emcee in total, 50 walkers, starting from the previously found best parameters, and 500 taking MCMC steps
my_model.run_emcee(2,16,50,best_parameters,500,output_filename='temporary_model')   





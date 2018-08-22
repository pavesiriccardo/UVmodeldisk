import numpy as np
from KinMS import KinMS
import uvutil
from astropy.io import fits
import pymultinest
from galario.double import sampleImage


from UVmodeldisk import uvmodeldisk  #import the main class


my_model=uvmodeldisk(6.,6.,4000.,.2,100.,'../data/gn10-C.statwt_selec.uvfits','../data/images/gn10-C.fits')  #Defines the model parameters, see class constructor

my_model.xpos_center_padded=126 #edit approximate center pixel for centering the KinMS model
my_model.ypos_center_padded=128

my_model.set_ellipt_gauss_continuum(-.012,-.068,.471,.471,0,8.)   #set the continuum 2D elliptical Gaussian model

#my_model.loglike([200.,.1,400.,.01,0,60.,180,0,0,10.])   #check that the lok likelihood works

#Define appropriate priors for each parameter

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

my_model.my_prior=prior   #Set the my_prior function of the object to point to the correct prior definition

my_model.run_Multinest(2)   #Run Multinest with 2 threads!

#execute this script with MPI by calling: mpiexec -n 15 python Class_testing_script_multinest.py


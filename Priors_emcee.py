import numpy as np
class Priors(object):
    """

    Usage e.g.:
        from priors import Priors
        pri=Priors()
        lnprior+=pri.UniformPrior(cube[0],1.0,199.0)
        etc.
    """

    def __init__(self):
        pass
    


    def UniformPrior(self,r,x1,x2):
        """Uniform[0:1]  ->  Uniform[x1:x2]"""
	if x1<=r<=x2:
	        return 0
	elif r>x2:
		return max(-1.0e32,-(r-x2)**2/2/(.01*(x2-x1))**2)
	else:
		return max(-1.0e32,-(r-x1)**2/2/(.01*(x2-x1))**2)

    def LogPrior(self,r,x1,x2):
        """Uniform[0:1]  ->  LogUniform[x1:x2]"""
	if x1<=r<=x2:
		return -np.log(r/x1)        
	elif r>x2:
		return max(-1.0e32,-np.log(x2/x1)-np.log(r/x2)**2/2/(.01*np.log(x2/x1))**2)
	else:
		return max(-1.0e32,-np.log(r/x1)**2/2/(.01*np.log(x2/x1))**2)


    def GaussianPrior(self,r,mu,sigma):
        """Uniform[0:1]  ->  Gaussian[mean=mu,variance=sigma**2]"""
        #from math import sqrt
        #from scipy.special import erfcinv
        #if (r <= 1.0e-16 or (1.0-r) <= 1.0e-16):
        #    return -1.0e32
        #else:
        #    return mu+sigma*sqrt(2.0)*erfcinv(2.0*(1.0-r))
	return -(r-mu)**2/2/sigma**2-0.5*np.log(2*np.pi*sigma**2)



    def  SinPrior(self,r,x1,x2):
        #Uniform[0:1]  ->  Sin[x1:x2]  (angles in degrees):
        #from numpy import cos,arccos
        deg2rad=np.pi/180.
        #cx1=cos(x1*deg2rad)
        #cx2=cos(x2*deg2rad)
        #return arccos(cx1+r*(cx2-cx1))/deg2rad
	if x1<=r<=x2:
		return np.log10(np.sin(r*deg2rad))
	elif r>x2:
		return max(-1.0e32,np.log10(np.sin(x2*deg2rad))-(r-x2)**2/2/(.01*(x2-x1))**2)
	else:
		return max(-1.0e32,np.log10(np.sin(x1*deg2rad))-(r-x1)**2/2/(.01*(x2-x1))**2)



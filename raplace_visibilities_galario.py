

import numpy as np,os
from KinMS import KinMS
import uvutil,sample_vis
from astropy.io import fits
from scipy.optimize import minimize
from galario.double import sampleImage


def replace_visibilities_galario(visdataloc,sbmodelloc,modelvisloc):
	#this needs to be open, channel by channel
	modelimage_cube = fits.getdata(sbmodelloc)
	#this can be the same throughout
	modelheader = fits.getheader(sbmodelloc)
	#these need to be peeled, channel by channel for the following
	uu_cube, vv_cube, ww_cube = uvutil.uvload(visdataloc)
	vis_complex_cube, wgt_cube = uvutil.visload(visdataloc)
	#this can be the same throughout
	pcd = uvutil.pcdload(visdataloc)
	#all the following is done channel by channel
	for chan in range(modelimage_cube.shape[0]):
		uu=np.ones((uu_cube.shape[0],uu_cube.shape[1],1,uu_cube.shape[3]))
		vv=np.ones((vv_cube.shape[0],vv_cube.shape[1],1,vv_cube.shape[3]))
		uu[:,:,0,:]=uu_cube[:,:,chan,:]
		vv[:,:,0,:]=vv_cube[:,:,chan,:]
		modelimage=modelimage_cube[chan]
		uushape = uu.shape
		if len(uushape) == 2:
        		npol = uushape[0]
        		nrow = uushape[1]
        		uushape = (npol, 1, nrow)
		uu = uu.flatten()
		vv = vv.flatten()
		uu=uu.copy(order='C')
		vv=vv.copy(order='C')
		modelimage=np.roll(np.flip(modelimage,axis=0),1,axis=0).copy(order='C').byteswap().newbyteorder()
		model_complex = sampleImage(modelimage, np.absolute(modelheader['CDELT1'])/180*np.pi, uu, vv)   # sample_vis.uvmodel(modelimage, modelheader, uu, vv, pcd)
		vis_complex = model_complex.reshape(uushape)
		vis_complex_cube[:,:,chan,:]=vis_complex[:,:,0,:]
		if chan%10==0:	
			print chan
	#modelvisloc='../replaced_visib.uvfits'
	os.system('rm -rf ' + modelvisloc[:-7]+'*')
	cmd = 'cp ' + visdataloc + ' ' + modelvisloc	
	os.system(cmd)
	real = np.real(vis_complex_cube)
	imag = np.imag(vis_complex_cube)
	visfile = fits.open(modelvisloc, mode='update')
	visibilities = visfile[0].data
	visheader = visfile[0].header
	if visheader['NAXIS'] == 7:
	            visibilities['DATA'][:, 0, 0, :, :, :, 0] = real
	            visibilities['DATA'][:, 0, 0, :, :, :, 1] = imag
	elif visheader['NAXIS'] == 6:
	            visibilities['DATA'][:, 0, 0, :, :, 0] = real
	            visibilities['DATA'][:, 0, 0, :, :, 1] = imag
	else:
	            print("Visibility dataset has >7 or <6 axes.  I can't read this.")
	#visibilities['DATA'][:, 0, 0, :, :, 2] = wgt
	visfile[0].data = visibilities
	visfile.flush()
	visfile.close()

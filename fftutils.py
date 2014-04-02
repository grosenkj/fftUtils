# Collection of function to calculate mag grids
import numpy as np, scipy.interpolate as sciint
def calcWavenumber(nrX,nrY,xD,yD):
	""" Function to calculate a derivatives in the Fourier domain.
	Returns kx, ky shifted wavenumbers 
	Inputs 
	nrX is number of data in x direction
	nrY is number of data in y direction
	xD is the grid interval in X direciton
	yD is the grid interval in Y direciton
	"""

	# Calculate the wavenumbers
	xval = np.fft.fftshift(np.fft.fftfreq(int(nrX),1./nrX))
	yval = np.fft.fftshift(np.fft.fftfreq(int(nrY),1./nrY))
	kxV = 2*np.pi*xval/(xD)
	kyV = 2*np.pi*yval/(yD)
	kx, ky = np.meshgrid(kxV,kyV)
	return kx,ky

def expand(grid):
	""" Function to expand/pad the grid to eliminate any edge structure.
	Inputs
		grid is the data grid.

	Outputs:
		gridE - expanded grid.
		gridInfo is a set of (lx1,lx2,ly1,ly2,sxE,syE):
			l{x/y}{1/2} number of padding cells in each direction.
			s{x/y}E in total number of cells used in the expanded grid
	"""
	# Size of the grid
	nx, ny = grid.shape
	# Find the average value of the edges of the grid.
	eave = np.average(np.concatenate((grid[[0,-1],:].ravel(),grid[1:-1,[0,-1]].ravel())))
	# Find the nearest power of 2 to the size of the grid for padding
	sxE, syE = np.power(2,np.ceil(np.log2(np.array([nx,ny]))))
	if sxE == nx: sxE = 2*sxE
	if syE == ny: syE = 2*syE
	# Calculate the number of padding cells
	lx1 = (sxE - nx)/2
	lx2 = sxE - nx - lx1
	ly1 = (syE - ny)/2
	ly2 = syE - ny - ly1
	# Calculate the padding values
	gridE = np.nan*np.zeros((sxE,syE),dtype=np.float)
	gridE[lx1:lx1+nx,ly1:ly1+ny] = grid
	pmask = np.isnan(gridE)
	# Make a psudo grid for populating the grid
	xG,yG = np.meshgrid(np.arange(sxE),np.arange(syE))
	gridE[pmask] = sciint.griddata((xG[~pmask].ravel(),yG[~pmask].ravel()),gridE[~pmask].ravel(),(xG[pmask].ravel(),yG[pmask].ravel()),'nearest')
	# Make a weight for mixing the data in the padding
	wx = np.nan*np.zeros(pmask.shape)
	wx[~pmask]=1
	wx[[0,-1],:]=0
	wx[:,[0,-1]]=0
	wmask = np.isnan(wx)
	wx[wmask] = sciint.griddata((xG[~wmask].ravel(),yG[~wmask].ravel()),wx[~wmask].ravel(),(xG[wmask].ravel(),yG[wmask].ravel()),'linear')
	
	# Mix the grids
	gridE = gridE*wx + eave*(1-wx)
	return gridE, (lx1,lx2,ly1,ly2,sxE,syE)

def extract(gridE,gridInfo):
	""" Function to expand/pad the grid to eliminate any edge structure.
	
	Input:
		gridE - expanded grid.
		gridInfo is a set of (lx1,lx2,ly1,ly2,sxE,syE):
			l{x/y}{1/2} number of padding cells in each direction.
			s{x/y}E in total number of cells used in the expanded grid
	Output:
		grid is the data grid.
	"""
	lx1,lx2,ly1,ly2,sxE,syE = gridInfo
	return gridE[lx1:sxE-lx2,ly1:syE-ly2]

# def reduceToPole(X,z,I,D):
# 	""" Function to calculate a reduction to pole.
# 	X are 2 coordinates matrices (from np.meshgrid, n x m)
# 	z is a grided values of the data, cooresponding to X coordinates
# 	I is the inclination 
# 	D is the declination
# 	"""

# 	# Fourier transform the data
# 	zexp,gInfo = expand(z)
# 	fft_z = np.fft.fftshift(np.fft.fft2(zexp))
# 	kx,ky = calcWavenumber(gInfo[4],gInfo[5],np.unique(np.diff(X[0]))*gInfo[4],np.unique(np.diff(X[0]))*gInfo[5])
# 	kAng = np.arctan(kx/ky)
# 	kAng[np.isnan(kAng)] = 0 
# 	# Calculate the RTP 
# 	fft_rtp = fft_z/(np.sin(np.deg2rad(I)) + 1j*np.cos(np.deg2rad(I))*np.cos(np.deg2rad(D)+kAng))**2
# 	# Take care of nan's
# 	# fft_rtp[np.isnan(fft_rtp)] = 1 + 1j
# 	return extract(np.fft.ifft2(np.fft.ifftshift(fft_rtp)),gInfo)

def calcThetaComp(kx,ky,I,D):
	""" Function to calculate the theta components of the fields.
	kx wavenumbers in x direction (East)
	ky wavenumbers in y direction (North)
	I inclination,
	D declination 

	"""
	kn = np.sqrt(kx**2 + ky**2) 
	knZ = kn == 0 
	theta = np.sin(np.deg2rad(I)) + ( 1j*( (np.cos(np.deg2rad(I))*np.cos(np.deg2rad(D)))*ky + (np.cos(np.deg2rad(I))*np.sin(np.deg2rad(D)))*kx ) / kn )
	theta[knZ] = 1
	return theta

def reduceToPole(X,z,I,D):
	""" Function to calculate a reduction to pole.
	X are 2 coordinates matrices (from np.meshgrid, n x m)
	z is a grided values of the data, cooresponding to X coordinates
	I is the inclination 
	D is the declination

	Based on Blakley, R.J., 1996. Petential Theory in Gravity adn Magnetic Application. Cambridge University Press.
	"""

	## Fourier transform the data
	# Expand the grid to deal with edge structures
	zexp,gInfo = expand(z)
	# FFT and shift to center
	fft_z = np.fft.fftshift(np.fft.fft2(zexp))
	# Calculate wavenumbers
	kx,ky = calcWavenumber(gInfo[4],gInfo[5],np.unique(np.diff(X[0]))*gInfo[4],np.unique(np.diff(X[0]))*gInfo[5])
	## Calculate the RTP 
	tm = calcThetaComp(kx,ky,I,D) 
	tf = calcThetaComp(kx,ky,I,D)
	rtp = 1/(tm*tf)
	fft_rtp = fft_z*rtp
	# Take care of nan's
	# fft_rtp[np.isnan(fft_rtp)] = 0 + 0j
	return extract(np.fft.ifft2(np.fft.ifftshift(fft_rtp)),gInfo)


def fftDerivatives(X,z):
	""" Fucntion to calculate spatial derivaties in the Fourier domain.
	X  are 2 coordinates matrices (from np.meshgrid, n x m)
	z is a grided values of the data, cooresponding to X coordinates
	"""
	# FFT the expanded grid
	zexp,gInfo = expand(z)
	fft_z = np.fft.fftshift(np.fft.fft2(zexp))
	kx,ky = calcWavenumber(gInfo[4],gInfo[5],np.unique(np.diff(X[0],axis=1))*gInfo[4],np.unique(np.diff(X[1],axis=0))*gInfo[5])
	fft_derX = 1j*kx*fft_z
	fft_derY = 1j*ky*fft_z
	# Take care of nan's
	# fft_rtp[np.isnan(fft_rtp)] = 1 + 1j
	return extract(np.fft.ifft2(np.fft.ifftshift(fft_derX)),gInfo), extract(np.fft.ifft2(np.fft.ifftshift(fft_derY)),gInfo)
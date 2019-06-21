#%%------------------------------------------------------------------------------
import sys, os, time, glob
from LightPipes import *
#import LightPipes as lp
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.signal import square
from matplotlib.colors import LinearSegmentedColormap

import matplotlib as mpl
import matplotlib.colors as colors
import colorcet as cc
from tqdm import tqdm
import joblib

from scipy.interpolate import splrep, sproot, splev, interp2d

sys.path.append()
import phaseGratingLP as pgLP

mpl.rcParams['figure.dpi'] = 240

my_cmap = mpl.cm.get_cmap('plasma')
my_cmap.set_under('w')
#%%
cmapy = LinearSegmentedColormap.from_list('mycmap', ['white','black', 'darkblue', 'darkred', 'dodgerblue'])
cmapy2 = cc.m_fire
class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):  
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

def fwhm(x, y, k=3):
    """
    Determine full-with-half-maximum of a peaked set of points, x and y.

    Assumes that there is only one peak present in the datasset.  The function
    uses a spline interpolation of order k.
    """

    half_max = np.amax(y)/2.0
    s = splrep(x, y - half_max, k=k)
    roots = sproot(s)

    if len(roots) > 2:
        raise MultiplePeaks("The dataset appears to have multiple peaks, and "
                "thus the FWHM can't be determined.")
    elif len(roots) < 2:
        raise NoPeaksFound("No proper peaks were found in the data set; likely "
                "the dataset is flat (e.g. all zeros).")
    else:
        return abs(roots[1] - roots[0])
	
#%%
def initializeBeamProfile(λ=1.35):
	dim = 20
	size=dim*mm
	wavelength=λ*um
	N=2**10
	R=10*mm
	mmppx = 20.0/N

	
	def func_int(i, j):
	    return np.exp(-0.5*((i - N/2)/(100.0*N/256.0))**2 - 0.5*((j - N/2)/(100.0*N/256.0))**2)
	def func_phase(i, j):
	    return 0.5*ζ*np.pi*(1+square(2.0*np.pi*((i-N/2)*mmppx - x0)/d))
	
#	xaxis = np.arange(N)
#	yaxis = np.arange(N)
#	Int = func_int(xaxis[:,None], yaxis[None,:])

#	N=2**10
	fname = 'C:/Data/Beam Profile/images/full_beam.dat'
	beam_img = np.loadtxt(fname)
	px_size = 0.0485 #[mm]
	x_width = 15.52
	y_width = 11.64
	x = px_size*np.arange(len(beam_img[0,:])) - x_width/2.0
	y = px_size*np.arange(len(beam_img[:,0])) - y_width/2.0
	x_f = x_width*np.arange(N)/N - x_width/2.0
	y_f = y_width*np.arange(N)/N - y_width/2.0
	beam_interp = interp2d(x,y,beam_img, kind='cubic')
	
	plot_img = False
	if plot_img:
		fig, ax = plt.subplots(1,2, sharex=True, sharey=True)
		ax[0].pcolormesh(x, y, beam_img, cmap='magma')
		ax[1].pcolormesh(x_f, y_f, beam_interp(x_f,y_f), cmap='magma')

	xd = dim*np.arange(N)/N - dim/2.0
	
	Int = beam_interp(xd, xd)
	Int[Int<0]=0.0
	return None

def focusIntensity(x0=0.0, ζ=1.0, d=2.5, λ=1.35, z_offset=0.0, lens_Fresnel=True):
	phase=[]
	i = np.arange(N)
	for j in np.arange(N):
		phase.append(0.5*ζ*np.pi*(1+square(2.0*np.pi*((i-N/2)*mmppx - x0)/d)))
	phase = np.transpose(phase)
	
	f=20*cm
	f1=4*m
	f2=f1*f/(f1-f)
	frac=f/f1
	newsize=frac*size
	
	F=Begin(size,wavelength,N)
	F=SubIntensity(Int,F)
	F=SubPhase(phase,F)
	F=CircAperture(R,0,0,F)
	#F=Fresnel(3.0*f,F)
	if lens_Fresnel:
		F2=Lens(f1,0,0,F)
		#F2=Lens(f,0,0,F)
		F2=LensFresnel(f2,f,F2)
		#F2=LensFresnel(f,f,F2)
		F2=Convert(F2)
		#F2 = Fresnel(z_offset*cm,F2)
	else:
		F2=Lens(f,0,0,F)
		F2=Fresnel(f - z_offset*cm,F2)
	phi2=np.array(Phase(F2))
	I2=np.array(Intensity(0,F2))
	x2=[]
	for i in range(N):
		x2.append((-newsize/2+i*newsize/N)/mm)
	return I2, phi2, I2[:,int(N/2)], phi2[:,int(N/2)], x2

#%%
I_lineout_rr = []
φ_lineout_rr = []
step = 1.0/5#0.025
start0 = time.time()
#itrr = np.arange(0.0,2.50+step,step)
itrr = np.arange(0.0,2+step,step)
#fname_I="intensity.dat"
#fname_φ="phase.dat"

initializeBeamProfile()

for z in tqdm(itrr):
	start = time.time()
	_, _, I_lineout, φ_lineout, χ = focusIntensity(x0=1.25/2, lens_Fresnel=True, ζ=z)
	I_lineout_rr.append(I_lineout)
	φ_lineout_rr.append(φ_lineout)
	end = time.time()
	elapsed = end - start
	#print("ζ =","%.2f"%z,", ","%.2f"%elapsed, "seconds ,", "%.2f"%(elapsed/60.0), "minutes")
	#print("x0 =","%.2f"%z,", ","%.2f"%elapsed, "seconds ,", "%.2f"%(elapsed/60.0), "minutes")
	#np.savetxt(fname_I, I_lineout_rr)
	#np.savetxt(fname_φ, φ_lineout_rr)
end = time.time()
elapsed = end - start0
print("Total elapsed time:", "%.2f"%elapsed, "seconds ,", "%.2f"%(elapsed/60.0), "minutes")

#%%
direc_I = 'C:/Users/Tebe/Dropbox/Py_SJH/Python_SJH_2/Two_Source/Phase Grating/intensity.dat'
direc_φ = 'C:/Users/Tebe/Dropbox/Py_SJH/Python_SJH_2/Two_Source/Phase Grating/phase.dat'

I = np.transpose(np.loadtxt(direc_I))#I_lineout_rr)
φ = np.transpose(np.loadtxt(direc_φ))#φ_lineout_rr)


direc_3I = 'C:/Users/Tebe/Dropbox/Py_SJH/Python_SJH_2/Two_Source/Phase Grating/intensity_5f.dat'
direc_3φ = 'C:/Users/Tebe/Dropbox/Py_SJH/Python_SJH_2/Two_Source/Phase Grating/phase_5f.dat'

I3 = np.transpose(np.loadtxt(direc_3I))#I_lineout_rr)
φ3 = np.transpose(np.loadtxt(direc_3φ))#φ_lineout_rr)
#%%
fig, ax = plt.subplots(2, sharex=True, sharey=True)
ax[0].imshow(I[1950:2050], aspect='auto', origin='lower',cmap=cc.m_fire, norm = colors.LogNorm(), clim=(2e2,1e5))#, clim=(2e2,1e5))
ax[1].imshow(I3[1950:2050], aspect='auto', origin='lower',cmap=cc.m_fire, norm = colors.LogNorm(), clim=(2e2,1e5))#
#%%
fig, ax = plt.subplots(2, sharex=True, sharey=True)
ax[0].imshow(φ[1950:2050], aspect='auto', origin='lower',cmap='twilight')
ax[1].imshow(φ3[1950:2050], aspect='auto', origin='lower',cmap='twilight')
#%%
fig, ax = plt.subplots()
ax.plot(np.unwrap(φ[2004])-np.unwrap(φ3[2004]), color='darkblue')
#ax.plot(np.unwrap(φ3[2004]-φ3[2004][0]), color='darkred')
#%%
fig, ax = plt.subplots()
ax.plot(I[2004,:],color='dodgerblue')
ax.plot(I3[2004,:],color='red')
ax2 = ax.twinx()
ax2.plot(np.unwrap(φ[2004,:]), color='darkblue')
ax2.plot(np.unwrap(φ3[2004,:]), color='darkred')
#%%
fig, ax = plt.subplots()
ax.semilogy(I[:,0],color='darkred')
#%%
fig, ax = plt.subplots()
ax.plot(I[:,49], label='-1')
#ax2 = ax.twinx()
#ax2.plot(np.unwrap(φ[:,49]), label='-1')
#ax.plot(I[1024,:]/np.amax(I[1024,:]),label='0')
#ax.plot(I[1246,:]/np.amax(I[1246,:]),label='1')
#%%
I_lf = np.transpose(I_lineout_rr)
φ_lf = np.transpose(φ_lineout_rr)
#%%
I_lf_9 = np.transpose(I_lineout_rr)
φ_lf_9 = np.transpose(φ_lineout_rr)
#%%
I_3lf = np.transpose(I_lineout_rr)
φ_3lf = np.transpose(φ_lineout_rr)
#%%
I_6lf = np.transpose(I_lineout_rr)
φ_6lf = np.transpose(φ_lineout_rr)
#%%
fig, ax = plt.subplots()#3, sharex=True, sharey=True)
ax.pcolormesh(itrr, χ, I_lf, cmap=cc.m_fire)
#ax[1].pcolormesh(np.arange(len(itrr)), np.arange(len(χ[360:440])), I_3lf[360:440], cmap='nipy_spectral')
#ax[2].pcolormesh(np.arange(len(itrr)), np.arange(len(χ[360:440])), I_6lf[360:440], cmap='nipy_spectral')
#ax[1].imshow(φ_lf, aspect='auto', origin='lower', cmap='nipy_spectral')
#%%
fig, ax = plt.subplots()
ax.plot(itrr,I_lf[402,:]/np.amax(I_lf[402,:]), '-o', color='darkblue',  alpha=0.7, label='0 f, -1')
ax.plot(itrr,I_lf[623,:]/np.amax(I_lf[623,:]), '-o', color='darkred',  alpha=0.7, label='0 f, +1')
ax2 = ax.twinx()
ax2.plot(itrr,I_lf[512,:]/np.amax(I_lf[512,:]), '-o', color='darkgreen',  alpha=0.7, label='0 f, 0')
#ax[1].plot(itrr,I_lf_9[402,:]/np.amax(I_lf_9[402,:]), '-o', color='darkblue',  alpha=0.7, label='0 f, -1, 0.9')
#ax[1].plot(itrr,I_lf_9[623,:]/np.amax(I_lf_9[623,:]), '-o', color='darkred',  alpha=0.7, label='0 f, +1, 0.9')
#ax.plot(itrr,I_3lf[402,:], '-o' ,color='darkred',  alpha=0.7, label='3 f')
#ax.plot(itrr,I_6lf[402,:], '-o' ,color='darkgreen',  alpha=0.7, label='6 f')
ax.legend(loc='best')
ax2.legend(loc='best')
#ax2 = ax.twinx()
#ax2.plot(χ, np.unwrap(φ_lf[:,1]),  color='dodgerblue',  alpha=0.3, label='0 f')
#ax2.plot(χ, np.unwrap(φ_3lf[:,1]),  color='red',  alpha=0.3, label='3 f')
#%%
a = I_lf[402,:]
#b = I_lf_9[402,:]
#c = I_6lf[402,:]

a_fft = np.abs(np.fft.fft(a))
#b_fft = np.abs(np.fft.fft(b))
#c_fft = np.abs(np.fft.fft(c))
ω = np.fft.fftfreq(len(a))
fig, ax = plt.subplots()
ax.plot(a_fft[1:], '-o', color='darkblue',  alpha=0.7, label='0 f')
#ax.plot(b_fft[1:], '-o' ,color='darkred',  alpha=0.7, label='0 f, 0.5')
#ax.plot(c_fft[1:], '-o' ,color='darkgreen',  alpha=0.7, label='6 f')
ax.legend(loc='best')
#ax.set_xlim(0,0.06)

#%%
fig, ax = plt.subplots()#3, sharex=True, sharey=True)
ax.imshow(np.abs(np.fft.fft(I_lf[360:640,:len(itrr)//2],axis=1)), cmap='jet', aspect='auto', origin='lower')
#ax[1].imshow(np.abs(np.fft.fft(I_3lf[360:440],axis=1))[:,1:], cmap='bone', aspect='auto', origin='lower')
#ax[2].imshow(np.abs(np.fft.fft(I_6lf[360:440],axis=1))[:,1:], cmap='gnuplot2', aspect='auto', origin='lower')








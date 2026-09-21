"""Circular azimuth features without leakage."""
import numpy as np

def harmonic(x, order=3):
 x=np.asarray(x,float); r=np.deg2rad(x); return np.concatenate([np.column_stack((np.sin(k*r),np.cos(k*r))) for k in range(1,order+1)],axis=1)
def circular_difference(a,b): return (np.asarray(b)-np.asarray(a)+180)%360-180
def bins(x,n=8): return np.floor((np.asarray(x)%360)/(360/n)).astype(int)

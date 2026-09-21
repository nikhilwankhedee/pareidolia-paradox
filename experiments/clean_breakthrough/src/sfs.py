"""Simple shape-from-shading/solar-aligned CPU descriptors."""
import numpy as np
def features(img,azimuth):
 x=np.asarray(img,float); gy,gx=np.gradient(x); a=np.deg2rad(float(azimuth)); proj=gx*np.cos(a)+gy*np.sin(a); perp=-gx*np.sin(a)+gy*np.cos(a)
 return np.asarray([x.mean(),x.std(),np.mean(np.abs(proj)),np.mean(np.abs(perp)),np.mean(proj**2),np.mean(perp**2),np.mean(proj>0),np.mean(perp>0)],np.float32)

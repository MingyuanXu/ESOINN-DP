import numpy as np
from TensorMol import *
def KineticEnergy(v,m):
    #v in A/fs
    #m in kg/mol
    return 0.5*np.dot(np.einsum('ia,ia->i',v,v)*pow(10.0,10.0),m)/len(m)

class Thermo:
    def __init__(self,m_,v_):
        """
        Velocity Verlet step with a Rescaling Thermostat
        """
        self.N = len(m_)
        self.m = m_.copy()
        self.T = PARAMS["MDTemp"]
        self.Teff = 0.001
        self.kT = IDEALGASR*pow(10.0,-10.0)*self.T # energy units here are kg (A/fs)^2
        self.tau = 30*PARAMS["MDdt"]
        self.name = "Rescaling"
        print("Using ", self.name, " thermostat at ",self.T, " degrees Kelvin")
        self.Rescale(v_)
        return

    def step(self,m,v,dt):
        self.Teff = (2./3.)*KineticEnergy(v,self.m)/IDEALGASR
        v *= np.sqrt(self.T/self.Teff)
        return v

    def Rescale(self,v_):
        # Do this elementwise otherwise H's blow off.
        for i in range(self.N):
                Teff = (2.0/(3.0*IDEALGASR))*pow(10.0,10.0)*(1./2.)*self.m[i]*np.einsum("i,i",v_[i],v_[i])
                if (Teff != 0.0):
                        v_[i] *= np.sqrt(self.T/(Teff))
        return

class Andersen(Thermo):
    def __init__(self,m,v):
        self.N = len(list(m))
        self.m = m.copy()
        self.T = PARAMS["MDTemp"]
        self.gamma=0.5
        self.name='Andersen'
        self.Teff = 0.001
        self.kT = IDEALGASR*pow(10.0,-10.0)*self.T # energy units here are kg (A/fs)^2
        self.tau = 30*PARAMS["MDdt"]
        print("Using ", self.name, " thermostat at ",self.T, " degrees Kelvin")
        self.Rescale(v)
         
    def step(self,m,v,dt):
        self.kT=IDEALGASR*pow(10.0,-10.0)*self.T
        s=np.sqrt(2.0*self.gamma*self.kT/self.m)
        for i in range(v.shape[0]):
            if (np.random.random() > self.gamma*dt):
                v[i]=np.random.normal(0.0,s[i],size=(3))
        return v


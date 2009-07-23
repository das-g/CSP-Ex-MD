#! /usr/bin/env python

# Lennard-Jones Liquid simulation
# (k) 2008 -- All rites reversed -- copy what you like
# Raphael Das Gupta ( raphaeld )

# See http://www.complexfluids.ethz.ch/CP/EXERCISES/cpp_LJ_MD.pdf for the task.
# Not implemented: Obervation of radial pair correlation function and pressure.

from __future__ import division
from numpy.linalg import norm
from numpy import array
from numpy import all # for bools in array
from numpy import arange
from numpy import fmod # C-Like modulo (different from python's %)
import random
from numpy import sqrt

# Parameters of the Simulation
# ============================
N = 100  # Number of Particles
duration = 1 # in seconds
dt = 0.005 # Timestep, in seconds 0.005


# Dimensionless LJ units and other Constants
# ==========================================
n = 0.8 # Particle number density
# (Temperature and Particle Mass are implicitly 1)
spacedimensions = 3
minimal_initial_particle_distance = 0.85 # in reduced LJ units


# Derived Quantities (all in reduced LJ units)
# ==================
V = N/n                    # Volume
s = V**(1/spacedimensions) # Side length of simulation box
s2 = s/2                   # Box will be [-s2,s2]^spacedimensions

def pbc_dist(a,b,halve_box_length):
    """
    returns the distance between a and b
    within a origin centered box considering
    periodic boundary conditions
    """
    return fmod(b-a,halve_box_length)

def ULJ(r,rcut=2.5):
    """
    Lennard-Jones Potential
    for two particles at distance r
    (cut off at rcut)
    """
    if r > rcut:
        return 0.0
    else:
        s=-4*(rcut**-12 - rcut**-6)
        return 4*(r**-12 - r**-6)+s

def FLJ(xlist,rcut=2.5):
    """
    Lennard-Jones Force
    returns a list of Forces
    for a List of coordinates
    """
    forcelist=[]
    for x in xlist:
        force = 0
        for dd in pbc_dist(x,xlist,s2): # traverse directed distance list
            d = norm(dd)
            if d > rcut or d == 0:
                pass
            else:
                force -= 24*(2*d**-13 - d**-7) * dd
        forcelist.append(force)
    return forcelist

class Statistics:
    def __init__(self):
        self.PE=[] # potential energies
        self.KE=[] # kinetic energies
    def sampleX(self,x):
        self.PE.append(sum([ULJ(norm(x1 - x2)) for x1 in x for x2 in x if not all(x1 == x2)]))
    
    def sampleV(self,v):
        self.KE.append(sum([norm(vel)**2 for vel in v])/2)

def currentTemperature(v):
    #script 6.39
    mvsq=0.0
    for vac in v:      
	for vcomp in vac:
            mvsq+=vcomp*vcomp
    mvsq/=N
    currentT=mvsq/(3.0*N)
    return currentT

def temperatureVScale(v):
    #script 6.42
    currentT=currentTemperature(v)    
    scale=sqrt(1./currentT)
    v*=scale
    #test
    #print "temperature: ", currentT 
    #print "scaled temperature: ", currentTemperature(v)  


def vv_step(x,v,a,dt,stat,F=FLJ,vScale=temperatureVScale):
    """
    Do one step of Velocity Verlet integration
    """
    x = fmod(x + v * dt + 0.5 * dt**2 * a,s2)
    stat.sampleX(x)  # accumulate x-dependent averages
    v += 0.5 * a * dt
    a = array(F(x))
    v += 0.5 * a * dt
    stat.sampleV(v)  # accumulate v-dependent averages
    vScale(v) # eventually rescale velocities


# Main Program:
# =============

print "Generating initial particle configuration:"
print "  * positions ...",
x=[]
while len(x)<N:
    particle = array( [random.uniform(-s2,s2) for d in range(spacedimensions)] )
    for other in x:
        if norm(fmod(particle-other,s2)) <= minimal_initial_particle_distance:
            break
    else:
        # left for loop without break ==> Particle isn't too near to any other
        x.append(particle)
x=array(x)
print "done"

print "  * velocities ...",
v=[]
for i in x:
    velocity = array( [random.gauss(0,1) for d in range(spacedimensions)] )
    v.append(velocity/norm(velocity))
v=array(v)
print "done"
print

print "SIMULATING ...",
a=array(FLJ(x))
stat=Statistics()
for t in arange(0,duration,dt):
    vv_step(x,v,a,dt,stat)
print "done"

print "Energies:"
print "Potential\t\tKinetic\t\tTotal"
print array( [ stat.PE, stat.KE, array(stat.PE)+array(stat.KE) ]).transpose()

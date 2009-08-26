#! /usr/bin/env python

# Lennard-Jones Liquid simulation
# (c) 2008-2009 Raphael Das Gupta
# All rites reversed -- copy what you like

# See http://www.ifb.ethz.ch/education/statisticalphysics/20090508_ex.pdf for the task.

from __future__ import division
from numpy.linalg import norm
from numpy import array
from numpy import all # for bools in array
from numpy import arange
from numpy import zeros_like,  zeros
from numpy import linspace, indices, ceil, column_stack
import random
from numpy import sqrt
from numpy import sum as npsum
import pylab
from sys import stdout

import cells

# Parameters of the Simulation
# ============================

# Dimensionless LJ units:
# sigma, particle_mass and eps are all implicitly 1

N = 27  # Number of Particles
duration = 0.1 # unit sigma*sqrt(particle_mass/eps)
dt = 0.5e-3 # Timestep, unit sigma*sqrt(particle_mass/eps)

n = 0.95 # Particle number density, unit particles per sigma^spacedimensions
spacedimensions = 3
#minimal_initial_particle_distance = 0.85 # unit sigma

samples_per_frame = int(0.2 / dt)

def fmod(numerator,  denominator):
    return ((numerator + denominator) % (2 * denominator)) - denominator

def pbc_dist(a,b,halve_box_length):
    """
    returns the distance between a and b
    within a origin centered box considering
    periodic boundary conditions (minimum
    image convention)
    """
    return fmod(b-a,halve_box_length)

def ULJ(r_squared,rcut=2.5):
    """
    Lennard-Jones Potential
    for two particles at distance r = sqrt(r_squared)
    (cut off at rcut)
    """
    if r_squared > rcut**2:
        return 0.0
    else:
        s=-4*(rcut**-12 - rcut**-6)
        return 4*(r_squared**-6 - r_squared**-3)+s

def FLJ(xlist, linked_cells = None, rcut=2.5):
    """
    Lennard-Jones Force
    returns a list of Forces
    for a List of coordinates
    """
    forcelist=[]
    if linked_cells is not None:
        linked_cells.distribute_positions(xlist) # pack positions into cells
    for x in xlist:
        force = zeros_like(x)
        if linked_cells is None:
            peers = xlist
        else:
            peers = linked_cells.get_near_positions(x)
        for dd in pbc_dist(x, peers, s2): # traverse directed distance list
            d = norm(dd)
            if d > rcut or d == 0:
                pass
            else:
                force -= 24*(2*d**-13 - d**-7) * dd
        forcelist.append(force)
    return forcelist

class Statistics:
    def __init__(self):
        self.PE=[] # potential energies, unit eps
        self.KE=[] # kinetic energies, unit eps
    
    def sampleX(self, x, linked_cells = None):
        if linked_cells is not None:
            potential_energy = 0.
            linked_cells.distribute_positions(x)
            for particle in x:
                peers = linked_cells.get_near_positions(particle)
                potential_energy += sum([ULJ(sum((particle - peer)**2))
                                         for peer in peers
                                         if not all(particle == peer)])
            self.PE.append(potential_energy)
        else:
            self.PE.append(sum([ULJ(sum((x1 - x2)**2))
                                for x1 in x for x2 in x
                                if not all(x1 == x2)]))
        global sample_nr, frame_nr, plot_points
        try:
            if sample_nr % samples_per_frame == 0:
                plot_points.set_xdata(x[:, 0]) # should be more efficiant than creating a new plot
                plot_points.set_ydata(x[:, 1])
                frame_nr += 1
                pylab.savefig("./%0*d.png" % (5,frame_nr))
            sample_nr += 1
        except NameError:
            plot_points, = pylab.plot(x[:, 0], x[:,  1], '.')
            sample_nr = 0
            frame_nr = 0
            pylab.savefig("./%0*d.png" % (5,frame_nr))
    
    def sampleV(self,v):
        self.KE.append(npsum(v**2)/2)
        # equivalent to but more efficiant than
        # self.KE.append(sum([norm(vel)**2 for vel in v])/2)

def currentTemperature(v):
    #script 6.39
    mvsq=npsum(v**2)
    mvsq/=N
    currentT=mvsq/(3.0*N)
    return currentT


def conserveVelocities(v):
     pass

def temperatureVScale(v):
    #script 6.42
    currentT=currentTemperature(v)    
    scale=sqrt(1./currentT)
    v*=scale
    #test
    #print "temperature: ", currentT 
    #print "scaled temperature: ", currentTemperature(v)  


def vv_step(x,v,a,dt,stat,linked_cells,F=FLJ,vScale=conserveVelocities):
    """
    Do one step of Velocity Verlet integration
    """
    x = fmod(x + v * dt + 0.5 * dt**2 * a,s2)
    stat.sampleX(x, linked_cells)  # accumulate x-dependent averages
    v += 0.5 * a * dt
    a = array(F(x,linked_cells))
    v += 0.5 * a * dt
    stat.sampleV(v)  # accumulate v-dependent averages
    vScale(v) # eventually rescale velocities


def initial_positions_random(N, n, min_distance, space_dim, dont_use_dim = 0):
    V = N/n                    # Usable volume, unit sigma^(space_dim - dont_use_dim)
    s = V**(1/(space_dim-dont_use_dim)) # Side length of simulation box, unit sigma
    s2 = s/2                   # Box will be [-s2,s2]^space_dim, so centered around the origin
    
    x=[]
    while len(x)<N:
        particle = array( [random.uniform(-s2,s2) for d in range(space_dim - dont_use_dim)] + [0 for d in range(dont_use_dim)] )
        for other in x:
            if norm(fmod(particle-other,s2)) <= min_distance:
                break
        else:
            # left for loop without break ==> Particle isn't too near to any other
            x.append(particle)
    x=array(x)
    return x, s2

def initial_positions_grid(N,  n,  space_dim,  dont_use_dim = 0):
    V = N/n                    # Usable volume, unit sigma^(space_dim - dont_use_dim)
    s = V**(1/(space_dim-dont_use_dim)) # Side length of simulation box, unit sigma
    s2 = s/2                   # Box will be [-s2,s2]^space_dim, so centered around the origin
    
    l = linspace(-s2, s2, num=ceil(N ** (1 / (space_dim - dont_use_dim))), endpoint=False)
    x = column_stack([l[index].flat for index in indices([len(l) for dim in range(space_dim - dont_use_dim)])])
    x = array(random.sample(x, N)) # we rounded up above, so let's only use N of the generated points.
    x = column_stack([x,  zeros([x.shape[0], dont_use_dim])])
    return x, s2

# Main Program:
# =============

print "Generating initial particle configuration:"
print "  * positions ...",; stdout.flush()
#x, s2 = initial_positions_random(N, n, minimal_initial_particle_distance, spacedimensions, 1)
x, s2 = initial_positions_grid(N, n, spacedimensions, 1)
print "done"

print "  * velocities ...",; stdout.flush()
v=[]
for i in x:
    velocity = array( [random.gauss(0,1) for d in range(spacedimensions - 1)] + [0] )
    v.append(0.8 * velocity / norm(velocity))
v=array(v)
print "done"
print

print "SIMULATING ...",; stdout.flush()
a=array(FLJ(x))
stat=Statistics()
lcells = cells.Cells(2.5,-s2,s2)
for t in arange(0,duration,dt):
    vv_step(x,v,a,dt,stat,lcells)
print "done"

print "Energies:"
print "Potential\t\tKinetic\t\tTotal"
print array( [ stat.PE, stat.KE, array(stat.PE)+array(stat.KE) ]).transpose()

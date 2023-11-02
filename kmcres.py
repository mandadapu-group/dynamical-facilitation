import numpy as np
import numba as nb
import time
import tqdm
import gsd.hoomd
import scipy.special

import sys
import pickle
import signal
import logging
import logging.handlers
log = logging.getLogger()

#Class based on: http://stackoverflow.com/a/21919644/487556
#This makes sure that you cannot interrupt an important part of the code
class DelayedInterrupt(object):
    def __enter__(self):
        self.signal_received = False
        self.old_handler = {}
        self.old_handler[signal.SIGINT] = signal.signal(signal.SIGINT, self.handler)
        self.old_handler[signal.SIGTERM] = signal.signal(signal.SIGTERM, self.handler)

    def handler(self, sig, frame):
        self.signal_received = (sig, frame)
        print(f'Signal {sig} received by the process. Delaying KeyboardInterrupt.')
    def __exit__(self, type, value, traceback):
        for sig in [signal.SIGINT, signal.SIGTERM]:
            signal.signal(sig, self.old_handler[sig])
            if self.signal_received:
                self.old_handler[sig](*self.signal_received)

#Vectorized Bessel functions
@nb.vectorize([nb.float64(nb.float64)])                                                                   
def bessel_vp(x):                                                                                        
    return scipy.special.j0(x)                                                                          
                                                                                                         
@nb.njit(nogil=True, fastmath=True)                                                                       
def j0_vp(x):                                                                                          
    return bessel_vp(x) 

#Bond angle order parameter
@nb.njit(fastmath=True)
def Theta(rsq,theta,psi,factor):
    #rsq =  x**2+y**2
    #theta = np.arctan2(x,y)
    return -factor*np.sin(2*theta-2*psi)/rsq

#Displacement vector field
@nb.njit(fastmath=True)
def ux(r,theta,psi,nu,newfactor):
    #r =  np.sqrtx**2+y**2
    #theta = np.arctan2(x,y)
    trigterm = (3-nu)*0.5*np.cos(theta-2*psi)+0.5*(1+nu)*np.cos(3*theta-2*psi)
    trigterm *= newfactor#np.pi*sigma**2/(4*2**0.5)
    return trigterm/r#sq

@nb.njit(fastmath=True)
def uy(r,theta,psi,nu,newfactor):
    #r =  np.sqrtx**2+y**2
    #theta = np.arctan2(x,y)
    trigterm = -(3-nu)*0.5*np.sin(theta-2*psi)+0.5*(1+nu)*np.sin(3*theta-2*psi)
    trigterm *= newfactor
    return trigterm/r#sq


#Compute the Boltzmann factor for prob to put/delete excitation
@nb.njit(fastmath=True)#,parallel=True)
def get_prob(psi,pos,exc_pos,exc_psi,v0):

    #Compute position difference
    dx = pos[0] - np.expand_dims(exc_pos[:,0], -1)
    dy = pos[1] - np.expand_dims(exc_pos[:,1], -1)
    rsq = dx**2 + dy**2

    #Compute the angle shifts
    thetai = np.arctan2(dy, dx)

    #Evaluate the energy barrier
    cos_term = np.cos(2*psi + 2*np.expand_dims(exc_psi, -1) - 4*thetai)
    val = np.sum(cos_term / rsq, axis=0)
    return np.exp(-v0* val)


#Average and normalize the Boltzmann factor over all possible angles
@nb.njit(fastmath=True)
def get_avgprob(pos,exc_pos,exc_psi,v0,samples): 
    # Compute the prob numerically
    psi = np.linspace(0, 2*np.pi,samples)
    pdf_vals = get_prob(psi,pos,exc_pos,exc_psi,v0)
    
    #Return the numerical integration
    val = np.sum(0.5 * (pdf_vals[1:] + pdf_vals[:-1]) * (psi[1:] - psi[:-1]))
    return val

#Sample random orientation according to the Boltzmann prob. via inverse transform method
@nb.njit(fastmath=True)#,parallel=True)
def sample_from_ppsi(pos,exc_pos,exc_psi,v0,num_samples,gridsamples):#,rng_state):
    # Compute the unnormalized probability distribution function (PDF) numerically
    psi = np.linspace(0, 2*np.pi, gridsamples)
    pdf_vals = get_prob(psi,pos,exc_pos,exc_psi,v0)
    
    # Compute the cumulative distribution function (CDF) numerically
    cdf_vals = np.zeros_like(psi)
    cdf_vals[1:] = np.cumsum(0.5 * (pdf_vals[1:] + pdf_vals[:-1]) * (psi[1:] - psi[:-1]))
    
    # Simpson's rule
    # a = psi[:-1]
    # b = psi[1:]
    # fa = pdf_vals[:-1]
    # fb = pdf_vals[1:]
    # m = (a + b) / 2
    # fm = np.interp(m, psi, pdf_vals)
    # intervals = (b - a) / 6 * (fa + 4*fm + fb)
    # cdf_vals[1:] = np.cumsum(intervals)
    
    cdf_vals /= cdf_vals[-1]
    
    # Sample from the CDF using inverse transform method
    u = np.random.ranf(size=num_samples)
    samples = np.interp(u, cdf_vals, psi)
    return samples

#Generate a triangular lattice
def generate_lattice(L, spacing):
    a = spacing 
    b = a*np.sqrt(3)/2 # distance between rows

    My = np.round(L/b).astype(int) # number of unit cells in y-direction
    Mx = np.round(L/a).astype(int)

    # Generate x-y coordinates of lattice points
    x = []
    y = []
    for i in range(Mx):
        for j in range(My):
            x.append(i*a + (j%2)*a/2)
            y.append(j*b)

    x = np.array(x)
    x -= np.mean(x)
    y = np.array(y)
    y -= np.mean(y)
    return np.vstack((x,y)).T#.tolist()

#Compute interaction energy. Vectorized in exc_pos but not in pos
@nb.njit(fastmath=True)
def get_energy(psi,pos,exc_pos,exc_psi,v0):

    #Compute position difference
    dx = pos[0] - np.expand_dims(exc_pos[:,0], -1)
    dy = pos[1] - np.expand_dims(exc_pos[:,1], -1)
    rsq = dx**2 + dy**2

    #Compute the angle shifts
    thetai = np.arctan2(dy, dx)

    #Evaluate the energy barrier
    cos_term = np.cos(2*psi + 2*np.expand_dims(exc_psi, -1) - 4*thetai)
    val = np.sum(cos_term / rsq, axis=0)
    return v0*val

#Compute interaction energy. Vectorized in pos but not in exc_pos
@nb.njit(fastmath=True)
def update_energy(psi,pos,exc_pos,exc_psi,v0):
    #Compute position difference
    dx = np.expand_dims(pos[:,0], -1)-exc_pos[0] 
    dy = np.expand_dims(pos[:,1], -1)-exc_pos[1]
    rsq = dx**2 + dy**2

    #Compute the angle shifts
    thetai = np.arctan2(dy, dx)

    #Evaluate the energy barrier
    cos_term = np.cos(2*psi + 2*exc_psi - 4*thetai)
    #print(np.shape(cos_term))
    return v0*cos_term / rsq

#Compute interaction energy when an excitation is present at the site
@nb.njit(fastmath=True)#,parallel=True)
def update_energy_del(psi,pos,exc_pos,exc_psi,v0):
    #Compute position difference
    dx = pos[:,0]-exc_pos[0] 
    dy = pos[:,1]-exc_pos[1]
    rsq = dx**2 + dy**2

    #Compute the angle shifts
    thetai = np.arctan2(dy, dx)

    #Evaluate the energy barrier
    cos_term = np.cos(2*psi + 2*exc_psi - 4*thetai)
    #print(np.shape(cos_term))
    return v0*cos_term / rsq

#Create a snapshot of the simulation
def create_snapshot_exc(N,L,position,orientation,typeid,spacing,weights,ux,uy,thetavals,activity,timestep):
    snap = gsd.hoomd.Snapshot()
    snap.particles.N = N
    snap.configuration.box = [L, L, 0, 0, 0, 0]
    snap.configuration.step = timestep
    snap.particles.position = np.vstack((position[:,0],position[:,1],np.zeros_like(position[:,0]))).T
    
    #HOOMD GSD file uses quarternion to sae orientation information. 
    #We'll just use the last component to save the orientation angle
    orient = np.zeros((N,4))
    orient[:,3] = orientation
    snap.particles.orientation = orient
   
    #Charge <--> bond-angle field
    snap.particles.charge = thetavals
    
    #Mass <--> Probability to put/delete an excitation
    snap.particles.mass = weights

    #TypeID <--> empty/occupied sites
    snap.particles.typeid = typeid.astype(int)
    snap.particles.types = ['A','B']
    
    #Diameter <--> persistence variable
    snap.particles.diameter = activity
    
    #Velocity <--> displacement field
    snap.particles.velocity = np.vstack((ux,uy,np.zeros_like(ux))).T
    return snap

class KMCModel:
    def __init__(self,J,nu,spacing,Rf,kappa,L,thetafactor,dispfactor):
        
        #Save and dump RNG state
        self.rng_state = np.random.get_state()
        ## System parameters. 
        self.J = J # beta*Jsigma
        self.nu = nu # Poisson's ratio
        self.fug = np.exp(-J) 
        
        self.v0 = kappa*J #interaction strength times beta*Jsigma
        self.factor = thetafactor # Prefactor for the bond angle field
        self.newfactor = dispfactor # Prefactor for the dispalcement field
        
        self.spacing = spacing #lattice spacing
        self.areaexc = 3**0.5/2*(0.5*self.spacing/Rf)**2 #the Jacobian factor for the probability to insert new excitations
        
        self.L = L # Linear system size
        self.meshpoints = generate_lattice(L, self.spacing) # The lattice grid to put excitations
        self.intpoints = generate_lattice(L, self.spacing) #Mesh for calculating the fields. We set to be equal to self.meshpoints for now
        self.Nmax = len(self.meshpoints)
        self.Npoints = len(self.intpoints)
        
        self.Npsi = np.round(J*50).astype(int) # Number of gridpoints for the orientation angle sampling (Npsi)
        if self.Npsi < 50:
            self.Npsi = 50
        self.psigrid = np.linspace(0,2*np.pi,self.Npsi) # Array of gridpoints for orientation angle
        self.energy = np.zeros((self.Nmax,self.Npsi+1),dtype=np.float64) # The total interaction energy array
        self.weights = np.ones(self.Nmax)*2*np.pi*self.areaexc #array of probability weights for each lattice point
        self.totalrate = np.sum(self.weights) #the total prob to produce excitations
        self.weights /= self.totalrate
        
        #Array storing time series of correlation function and number of excitations
        
        #Fields
        self.thetavals = np.zeros(len(self.intpoints)) #bond angle field
        self.ux = np.zeros(len(self.intpoints),dtype=float) #x-comp of displacement field 
        self.uy = np.zeros(len(self.intpoints),dtype=float) #y-comp of displacement field
        self.persist = np.zeros(len(self.intpoints),dtype=int) #frame.particles.typeid.astype(int))
        
        #Auto-Correlation functions
        self.Cb = 1 #bond-order correlation function
        self.Fsk = 1 #Self-intermediate scattering function
        self.Pt = 1 #Persistence function
        self.MSD = 0 #elastic MSD
        self.tmarkov = 0.0 #Current time
        self.N = 0 #Number of excitations

        self.name = None
        
        self.file = None
        self.save = False
        
        self.gsdfile = None
        self.gsdfilerestart = None
        self.savegsd = False 
        
        self.reachth = False
        self.period = 1
        self.step = 0

        self.occupation = np.zeros(self.Nmax,dtype=bool) #Array of occupation variables
        self.excpsi = -np.ones(self.Nmax,dtype=np.float64)  #Array of orientation angle of occupied sites (-1, if not occupied)
        
    #Create files that save configurations and observables
    def create_logfile(self, name,save=False,savegsd=False):
        self.name = name 
        self.save = save
       
        with open('{}_rng_state.pkl'.format(name), 'wb') as f:
            pickle.dump(self.rng_state, f)
        
        #Log file, storing averaged observables
        self.file = open(name+".log",'w')
        self.file.write("timestep tmarkov Cb Fsk MSD Pt N threshold \n")
        self.log_results()
       
        #GSD file, storing configurations and fields
        self.savegsd = savegsd
        if self.savegsd:
            self.gsdfile = gsd.hoomd.open("{}.gsd".format(name),"wb")
            snap = create_snapshot_exc(self.Nmax,self.L,self.meshpoints,self.excpsi,self.occupation,self.spacing,self.weights,self.ux,self.uy,self.thetavals,self.persist,self.step)
            self.gsdfile.append(snap)
        
        #Restart GSD file
        self.gsdfilerestart = gsd.hoomd.open("{}_restart.gsd".format(name),"wb")
        snap = create_snapshot_exc(self.Nmax,self.L,self.meshpoints,self.excpsi,self.occupation,self.spacing,self.weights,self.ux,self.uy,self.thetavals,self.persist,self.step)
        self.gsdfilerestart.append(snap)
    
    #Save the averaged observables into log file and dump interaction energy data
    def log_results(self):
        self.file.write("{} {} {} {} {} {} {} {} \n".format(self.step,self.tmarkov,self.Cb,self.Fsk,self.MSD,self.Pt,self.N,self.reachth))
        self.file.flush()
        
        np.save('{}_deltaE'.format(self.name),self.energy)
        with open('{}_rng_state.pkl'.format(self.name), 'wb') as f:
            pickle.dump(self.rng_state, f)
    
    #Load log and GSD files. Used to restart the simulation 
    def load_logfile(self,filename,savegsd=False):
        self.name = filename
        with open('{}_rng_state.pkl'.format(self.name), 'rb') as f:
            self.rng_state = pickle.load(f)
        np.random.set_state(self.rng_state)
        
        #Load the log files and update
        with open('{}.log'.format(self.name),"r") as f:
            for line in f:
                pass
            last_line = line.split()
        self.file = open('{}.log'.format(self.name),'a+')
        self.step = float(last_line[0])
        self.tmarkov = float(last_line[1])
        self.Cb = float(last_line[2]) #bond-order correlation function
        self.Fsk = float(last_line[3]) #Self-intermediate scattering function
        self.MSD = float(last_line[4])
        self.Pt = float(last_line[5]) #Persistence function

        self.N = float(last_line[6])
        self.reachth = last_line[7] == 'True'
        
        self.save = True
        self.savegsd = savegsd
        
        #Load the gsdfile files and update the fields 
        if self.savegsd:
            self.gsdfile = gsd.hoomd.open("{}.gsd".format(self.name),"rb+")
        self.gsdfilerestart = gsd.hoomd.open("{}_restart.gsd".format(self.name),"rb+")
        snap = self.gsdfilerestart[-1]
        
        #Exc. DOFs
        self.occupation = snap.particles.typeid.astype(bool)
        self.excpsi = snap.particles.orientation[:,3]
        
        #Ordparam fields
        self.thetavals = snap.particles.charge
        self.thetavals.setflags(write=1)
        self.ux = snap.particles.velocity[:,0]
        #self.ux.setflags(write=1)
        self.uy = snap.particles.velocity[:,1]
        #self.uy.setflags(write=1)
        self.persist = snap.particles.diameter
        self.persist.setflags(write=1) 
        self.weights = snap.particles.mass
        self.weights.setflags(write=1)
        self.weights /= np.sum(self.weights)
        
        #newweights = np.zeros_like(self.weights)
        
        #Recompute the energies!
        self.energy = np.load('{}_deltaE.npy'.format(self.name))

    def move(self):
        #Update time
        dtmarkov = 1/(self.fug*self.totalrate)*np.log(1/np.random.uniform())
        self.tmarkov += dtmarkov
        
        #Choose random position
        idx = np.random.choice(np.arange(0,self.Nmax), 1, p=self.weights)[0].astype(int)
        newx = self.meshpoints[idx]

        insert = False
        #Inserting a new excitation
        if not self.occupation[idx]:#idx in self.empty_id:
            #Sample a new orientation angle
            if self.occupation.any():
                #If there are any occupied sites, we sample via proper distribution
                newpsi = sample_from_ppsi(newx,
                                          self.meshpoints[self.occupation],
                                          self.excpsi[self.occupation],
                                          self.v0,1,self.Npsi)[0]
            else:
                #Otherwise we know that the angle is equiprobable
                newpsi = np.random.uniform(0,2*np.pi)
            self.N += 1
            insert = True
            
        #Deleting an excitation
        else:  
            newpsi = self.excpsi[idx]
            self.N -= 1
        
        #Recompute the weights and totalprob
        self.compute_weights(insert,idx,np.array(newx),newpsi,self.psigrid,self.areaexc,
                            self.meshpoints,self.occupation,self.excpsi,
                            self.weights,self.energy,self.v0)
       
        #Compute observables
        dx = self.intpoints[:,0]-newx[0]
        dy = self.intpoints[:,1]-newx[1]
        self.Cb, self.MSD, self.Fsk = self.update_ordparam(dx,dy,self.spacing,newpsi,
                            self.ux,self.uy,self.thetavals,
                            self.factor,self.newfactor,insert,self.nu)#:
        exc_index = self.occupation > 0
        index = self.persist < 1
        realindex = exc_index & index
        self.persist[realindex] = 1
        self.Pt = 1-np.mean(self.persist)
        
    
    @staticmethod
    @nb.njit(nogil=True,fastmath=True)#nopython=True)  
    def update_ordparam(dx,dy,spacing,newpsi,Ux,Uy,thetavals,factor,newfactor,insert,nu):
        #Compute the bond orientation order parameter
        rsq = dx**2+dy**2
        index = rsq < 0.25*spacing**2
        theta = np.arctan2(dy,dx) 
        rsq[index] = 0.25*spacing**2
        rs = np.sqrt(rsq)
        if insert:
            thetavals += Theta(rsq,theta,newpsi,factor)
            
            Ux += ux(rs,theta,newpsi,nu,newfactor)
            Uy += uy(rs,theta,newpsi,nu,newfactor)
        else:
            thetavals -= Theta(rsq,theta,newpsi,factor)
            
            Ux -= ux(rs,theta,newpsi,nu,newfactor)
            Uy -= uy(rs,theta,newpsi,nu,newfactor)
        
        #Compute all relaxation measures
        Cb = np.mean(np.cos(6*thetavals))
        val = Ux**2+Uy**2
        MSD = np.mean(val)
        Fsk = np.mean(j0_vp(2*np.pi*np.sqrt(val)))
        return Cb, MSD, Fsk
    
    @staticmethod
    @nb.njit(nogil=True,fastmath=True)#nopython=True)  
    def compute_weights(insert,idx,newx,newpsi,psigrid,areaexc,
                        meshpoints,occupation,excpsi,weights,energy,v0):
        if insert:
            #Update excitation site energies
            if occupation.any():
                #Update old excitation sites w/ vectorization O(1) 
                energy[occupation,-1] += update_energy_del(excpsi[occupation],
                                                                meshpoints[occupation],
                                                                newx,newpsi,v0) 
                #Update the new excitation site independently w/ vectorization O(1)
                energy[idx] = get_energy(newpsi,newx,
                                        meshpoints[occupation],
                                        excpsi[occupation],v0)
            else:
                energy[idx] = 0.0
            
            #Update probs, occupation variables, and orientation angle
            weights[occupation] = np.exp(-energy[occupation,-1])
            occupation[idx] = True
            excpsi[idx] = newpsi
            
            #Update energies for all empty sites w/ vectorization O(1)
            energy[~occupation,:-1] += update_energy(psigrid,meshpoints[~occupation],
                                                        newx,newpsi,v0)
            probins = areaexc*np.exp(-energy[~occupation,:-1])
            #probins = np.exp(-energy[~occupation,:-1])
            weights[~occupation] = np.sum(0.5 * (probins[:,1:] + probins[:,:-1]) * (psigrid[1:] - psigrid[:-1]),axis=1)
       
        else:
            #Update energies for all old empty sites w/ vectorization O(1)
            energy[~occupation,:-1] -= update_energy(psigrid,meshpoints[~occupation],
                                                        newx,newpsi,v0)
            
            #Update the new empty site independently w/ vectorization O(1)
            occupation[idx] = False
            excpsi[idx] = -1.0
            if occupation.any():
                energy[idx,:-1] = get_energy(psigrid,newx,
                                                    meshpoints[occupation],
                                                    excpsi[occupation],v0)
            else:
                energy[idx,:-1] = 0.0
           
            #Update probs of empty sites
            probins = areaexc*np.exp(-energy[~occupation,:-1])
            weights[~occupation] = np.sum(0.5 * (probins[:,1:] + probins[:,:-1]) * (psigrid[1:] - psigrid[:-1]),axis=1)
            
            
            #Update excitation site energies
            if occupation.any(): 
                energy[occupation,-1] -= update_energy_del(excpsi[occupation],
                                                                        meshpoints[occupation],
                                                                        newx,newpsi,v0) 
                weights[occupation] = np.exp(-energy[occupation,-1])
        totalrate = np.sum(weights)
        weights /= totalrate
        
    #Functino to run the simulation
    def run(self,finalT=1,period=10,verbose=True,Pt_break=-np.inf,Fsk_break=-np.inf,Cb_break=-np.inf,consolefile=None,initstep=0):
        #finalT is units of Monte Carlo sweeps (moves per lattice site)
        maxT = int(finalT*self.Nmax) #Maximum timestep
        if maxT < 1:
            maxT = 1
        
        if consolefile:
            progressbar = tqdm.tqdm(range(initstep,maxT),file=consolefile)
        else:
            progressbar = tqdm.tqdm(range(initstep,maxT))
        
        for i in (progressbar if verbose else range(initstep,maxT)): 
            #Perform a single Monte Carlo move
            self.move()
            with DelayedInterrupt():
                #Periodically save results
                self.step += 1   
                if i % period == 0 and self.save:
                    #print(np.sum(self.energy))
                    self.log_results()
                    if self.savegsd:
                        snap = create_snapshot_exc(self.Nmax,self.L,self.meshpoints,self.excpsi,self.occupation,self.spacing,self.weights,self.ux,self.uy,self.thetavals,self.persist,self.step)
                        self.gsdfile.append(snap)
                    snap = create_snapshot_exc(self.Nmax,self.L,self.meshpoints,self.excpsi,self.occupation,self.spacing,self.weights,self.ux,self.uy,self.thetavals,self.persist,self.step)
                    self.gsdfilerestart.truncate()
                    self.gsdfilerestart.append(snap)
               

                #If all sites are occupied, stop the simulation!
                if self.occupation.all():##len(self.empty_pos) == 0:
                    print("There are no more empty sites!")
                    self.log_results()
                    if self.savegsd:
                        snap = create_snapshot_exc(self.Nmax,self.L,self.meshpoints,self.excpsi,self.occupation,self.spacing,self.weights,self.ux,self.uy,self.thetavals,self.persist,self.step)
                        self.gsdfile.append(snap)
                    snap = create_snapshot_exc(self.Nmax,self.L,self.meshpoints,self.excpsi,self.occupation,self.spacing,self.weights,self.ux,self.uy,self.thetavals,self.persist,self.step)
                    self.gsdfilerestart.truncate()
                    self.gsdfilerestart.append(snap)
                
                #If Fsk(t) reaches threshold, stop the simulation!
                if self.Fsk < Fsk_break:
                    self.reachth = True
                    self.log_results()
                    if self.savegsd:
                        snap = create_snapshot_exc(self.Nmax,self.L,self.meshpoints,self.excpsi,self.occupation,self.spacing,self.weights,self.ux,self.uy,self.thetavals,self.persist,self.step)
                        self.gsdfile.append(snap)
                    snap = create_snapshot_exc(self.Nmax,self.L,self.meshpoints,self.excpsi,self.occupation,self.spacing,self.weights,self.ux,self.uy,self.thetavals,self.persist,self.step)
                    self.gsdfilerestart.truncate()
                    self.gsdfilerestart.append(snap)
                
                #If Cb(t) reaches threshold, stop the simulation!
                if self.Cb < Cb_break:
                    self.reachth = True
                    self.log_results()
                    if self.savegsd:
                        snap = create_snapshot_exc(self.Nmax,self.L,self.meshpoints,self.excpsi,self.occupation,self.spacing,self.weights,self.ux,self.uy,self.thetavals,self.persist,self.step)
                        self.gsdfile.append(snap)
                    snap = create_snapshot_exc(self.Nmax,self.L,self.meshpoints,self.excpsi,self.occupation,self.spacing,self.weights,self.ux,self.uy,self.thetavals,self.persist,self.step)
                    self.gsdfilerestart.truncate()
                    self.gsdfilerestart.append(snap)
                
                #If P(t) reaches threshold, stop the simulation!
                if self.Pt < Pt_break:
                    self.reachth = True
                    self.log_results()
                    if self.savegsd:
                        snap = create_snapshot_exc(self.Nmax,self.L,self.meshpoints,self.excpsi,self.occupation,self.spacing,self.weights,self.ux,self.uy,self.thetavals,self.persist,self.step)
                        self.gsdfile.append(snap)
                    snap = create_snapshot_exc(self.Nmax,self.L,self.meshpoints,self.excpsi,self.occupation,self.spacing,self.weights,self.ux,self.uy,self.thetavals,self.persist,self.step)
                    self.gsdfilerestart.truncate()
                    self.gsdfilerestart.append(snap)

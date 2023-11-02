from kmcres import *
nb.set_num_threads(1)
print("Threading layer -- ",nb.threading_layer())
import os, glob
import sys

Filename = 'exc'

size = float(sys.argv[1]) #linear system size
J = float(sys.argv[2]) #betaJsigma
kappa = float(sys.argv[3]) #interaction strength
ld = float(sys.argv[4]) #lattice spacing
ktype = int(sys.argv[5]) #k-th replica run
gsdlimit = int(sys.argv[6]) #index of replica that will output GSD files


#Parameters at the tranision state
nu = 0.36 #Poisson's ratio
udd = 0.116 #u double dagger at transition state
ec = 2*2**0.5*(udd*(1+udd))/(1+2*udd) #eigenstrain

#Parameters after crossing transition state
Rexc = 3**0.5/2.0 #Final excitation radius
ef = 2*ec #Final eigenstrain
Rf = Rexc

#Prefactors
factor = (2**0.5)*(Rf**2)*ef/(1+nu) #prefactor for bond angle field
dispfactor = (2*Rf)**2/(4*2**0.5*(1+nu))*ef #prefactor for displacement field
kappatild = kappa*(2*Rf)**2 #the interaction strength to be inputted to model needs excitation diameter!

#Threshold on when you should stop simulation. Choose when overall persistence variable reach below 0.05 for now
th = 0.05
if not os.path.exists('{}'.format(ktype+1)):
    os.makedirs('{}'.format(ktype+1))

print("---- L =",size," ----")
print("---- ld =",ld," ----")
print("---- βJ =",J," ----")
print("---- κ =",kappa," ----")

if not os.path.exists('{}'.format(ktype+1)):
    os.makedirs('{}'.format(ktype+1))

fh = open('{}/progress.txt'.format(ktype+1), 'w')
name = "{}/{}".format(ktype+1,Filename)

#Here's a restarting scheme
#Check if a text file denoting run finished exists. If it does, then we can skip simulation
if os.path.exists('{}_finished.txt'.format(name)):
    print("Run already finished!")
else:
    #Create kMC simulation object
    kmc = KMCModel(J,nu,ld,Rf,kappatild,L=size*ld,thetafactor=factor,dispfactor=dispfactor)
   
    #Set number of steps in Monte Carlo Sweep units
    finalT = 10**6
    
    #Let's set sampling period to 1 Monte Carlo Sweep
    period = int(kmc.Nmax)
   
    #If restart file exists, we have to load them
    if os.path.exists('{}_restart.gsd'.format(name)):
        kmc.load_logfile(name,savegsd=True)
        initstep = kmc.step
        kmc.run(finalT=finalT,period=period,Pt_break=th,verbose=True,initstep=initstep)
    else:
        kmc.create_logfile(name,save=True,savegsd=True)
        kmc.run(finalT=finalT,period=period,Pt_break=th,verbose=True)
    
    f = open("{}_finished.txt".format(name),"w")
    f.write("Run finishes!")
    f.close()
    del f
    del kmc

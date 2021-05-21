# Header file for JILA eEDM simulation v7.1
# Last modified 01.04.2021

import os

if __name__ != "__main__":
    print("Importing required modules and functions from "+os.path.basename(__file__)+" ...")


# Import libraries
from qutip import *
from qutip.parallel import parfor
from numpy import *
from numpy import random
from pylab import *
from mpl_toolkits.mplot3d import Axes3D

from numpy import linalg as LA # diagonalization

from matplotlib.patches import ConnectionPatch # cross subplot annotation

from scipy.integrate import solve_ivp # to solve kinematics
from scipy.special import ellipk, ellipe # for B-field due to coil
from scipy.special import factorial # for Franck-Condon (not main class)
from scipy.special import mathieu_a,mathieu_b # for Mathieu parameters (not main class)
from scipy.io import loadmat # import matlab binary file
from scipy.optimize import curve_fit # for curve fitting
from scipy.interpolate import RegularGridInterpolator,interp1d,interp2d # for interpolation
from scipy.linalg import expm # for matrix exponential (not main class)

from sympy.physics.wigner import wigner_3j,wigner_6j,clebsch_gordan
import sympy as sp # for N(), also need ~`.astype(float64)` after `array`

import pickle,gzip # save state
import datetime # for generating filename for saving state

import pandas as pd # for easy data visualization


####------------ Object class ------------####
class sim:
    'Ramsey Fringe Simulation'
    
    # Class construction
    def __init__(self,minJ=1,maxJ=1,Joffset=None,HrotSwitchBool=True,boost=None,XF=None,period_num=None,dampBool=None,silent=None,f_exp=None,t_step=None,loadTrap=None):
        
        if silent==None:
            print("Initializing...")
        
        # Populate constants
        if silent==None:
            print("\t Generating constants... ",end='')
        if XF==None:
            self.k = k_Hf()
            if silent==None:
                print("Using HfF+... ",end='')
        else:
            self.k = k()
            if silent==None:
                print("Using ThF+... ",end='')
        if silent==None:
            print("Done!")
        
        # Populate parameters
        if silent==None:
            print("\t Generating parameters... ",end='')
        self.p = p(kk=self.k,boost=boost,period_num=period_num,f_exp=f_exp,t_step=t_step,loadTrap=loadTrap)
        self.gen_Mathieu()
        if silent==None:
            print("Done!")
        
        # Generate time steps
        if silent==None:
            print("\t Generating time steps... ",end='')
        self.p.gen_time_steps()
        if silent==None:
            print("Done!")
        
        # Generate quantumness
        if silent==None:
            print("\t Generating quantumness... ",end='')
        self.q = q(minJ,maxJ,Joffset,HrotSwitchBool,handle=self)
        if silent==None:
            print('Done!')
        
        self.fit_plots = True # whether to fit the plots for states
        self.fitted = False # whether the states are fitted
        self.fit_motion_plots = True # whether to fit the plots for motion
        self.fitted_motion = False # whether motions are fitted
        self.ested = False # whether {∆,d∆,g,dg} are estimated
        self.ested_Para = False # whether {∆,d∆,∆mF,d∆mF} are estimated
        
        # Damping coefficient to damp out slosh and ion temperature
        self.damp_b = 0
        if dampBool==None:
            self.dampBool = False
        else:
            self.dampBool = dampBool

    # Generate motion of ion in trap
    def gen_motion(self):
        
        datestring = datetime.datetime.now()
        ds = datestring.strftime("%H:%M:%S")
        print('('+ds+'): Generating pre-motion...')
        
        self.gen_Mathieu()
        
        # Initial [vx vy vz x y z]
        y0 = [ array([self.p.v0 , array([0,0,0]) ]).flatten() for Rswitch in range(2) ]
        
        # Turn on ramp to evaluate ramping up part
        self.p.E_rot_ramp = True
        self.p.E_displace_ramp = True
        
        # Whether to damp motion
        if self.dampBool:
            self.damp_b = 1000 # balance bv and eE/m so that z components have the same order of magnitude
        else:
            self.damp_b = 0

        # Determine commensurate cut
        """
            Need to make sure that the fields look exactly the same at cut and at beginning.
            Check commensuration of f_rot and f_rf (and whatever else that matters)
        """
        cut = int( 1/self.p.t_step / gcd.reduce([ int(self.p.f_rot) , int(self.p.f_rf) , int(self.p.f_dit) ]) )
        if self.p.time_steps[cut] < self.p.E_rot_ramp_time:
            cut = int( ceil(self.p.E_rot_ramp_time / self.p.time_steps[cut+1]) * cut )
        
        if len(self.p.time_steps) > cut:
            self.p.kine_time_steps = self.p.time_steps[:cut+1+1] # first +1 is for python, second +1 is for first element of next cycle
        else:
            self.p.kine_time_steps = arange(0,self.p.t_step*(cut+1+1),self.p.t_step)

        # Solving Kinematics ODE, result from odeint is in [ vx[t] vy[t] vz[t] x[t] y[t] z[t] ]
        [self.simul_v_r_p,self.simul_v_r_m] = parfor(kineSolve,[1,-1],kwarg={'self':self, 'y0':y0, 'kine_time_steps':self.p.kine_time_steps})
        self.simul_v = [self.simul_v_r_p[0:3],self.simul_v_r_m[0:3]]
        self.simul_r = [self.simul_v_r_p[3:6],self.simul_v_r_m[3:6]]
        
        v0 = [self.simul_v[0][:,cut],self.simul_v[1][:,cut]]
        r0 = [self.simul_r[0][:,cut],self.simul_r[1][:,cut]]
        
        self.simul_vblah = copy(self.simul_v)
        self.simul_rblah = copy(self.simul_r)
        self.p.kine_time_stepsblah = copy(self.p.kine_time_steps)
        
        y0 = [ array([v0[Rswitch] , r0[Rswitch]]).flatten() for Rswitch in range(2) ]

        datestring = datetime.datetime.now()
        ds = datestring.strftime("%H:%M:%S")
        print('('+ds+'): Generating motion...')
        
        # Turn off ramp to evaluate ramped part
        self.p.E_rot_ramp = False
        self.p.E_displace_ramp = False
        self.damp_b = 0
        
        # For solving actual trajectory
        self.p.kine_time_steps = self.p.time_steps

        # Solving Kinematics ODE, result from odeint is in [ vx[t] vy[t] vz[t] x[t] y[t] z[t] ]
        [self.simul_v_r_p,self.simul_v_r_m] = parfor(kineSolve,[1,-1],kwarg={'self':self, 'y0':y0, 'kine_time_steps':self.p.kine_time_steps})
        self.simul_v = [self.simul_v_r_p[0:3],self.simul_v_r_m[0:3]]
        self.simul_r = [self.simul_v_r_p[3:6],self.simul_v_r_m[3:6]]
        
        self.p.ts_check = False # indicate no need to regenerate motion

        datestring = datetime.datetime.now()
        ds = datestring.strftime("%H:%M:%S")
        print('('+ds+'): Done!')
        
        self.print_motion()

    # Calculate trajectory parameters
    def print_motion(self):
        
        print('----')
        
        Rswitch_text = ['+1','-1']
        xyz_text = ['x','y','z']
        
        self.simul_rpp = [ [ [] for n in range(3) ] for n in range(2) ]
        self.simul_rmean = [ [ [] for n in range(3) ] for n in range(2) ]
        
        for Rswitch in range(2):
            for xyz in range(3):
                self.simul_rpp[Rswitch][xyz] = max(self.simul_r[Rswitch][xyz])-min(self.simul_r[Rswitch][xyz])
                self.simul_rmean[Rswitch][xyz] = mean(self.simul_r[Rswitch][xyz])
                
                print('\t'+xyz_text[xyz]+'[R='+Rswitch_text[Rswitch]+']: Peak-to-peak = %.3g , Mean = %.3g'%(self.simul_rpp[Rswitch][xyz] , self.simul_rmean[Rswitch][xyz]))
            print('----')
    
    # Check if motion trajectory is generated
    def motion_check(self):
        if self.p.ts_check: # check if motion needs to be (re)generated
            print("Trajectory was not computed. Computing now...")
            self.gen_motion()
            print("Done!")
    
    # Check if Schrodinger's equation is solved
    def state_check(self):
        if not self.p.ss_check: # check if state needs to be (re)solved
            print("Quantum evolution not computed. Computing now...",end='')
            self.solve_SE()
            print("Done!")
    
    # Mathieu parameters
    def gen_Mathieu(self,output_Bool=None):
        """
            From my own working,
                ax = -|a0| + a'
                ay = -|a0| - a'
                az = 2|a0|
            where
                a0 = - 4 Q kappa U0 / ( m (2 pi f_rf)^2 Z0^2 )
                a' = 8 Q V' / ( m (2 pi f_rf)^2 R_eff^2 )
            and
                qx = -q0
                qy = q0
                qz = 0
            where
                q0 = 2 Q Vrf,pp / ( m (2 pi f_rf)^2 R_eff^2 )
        """
        self.p.trap_mathieu_a0 = -4 * (self.k.fund_e * self.k.XF_charge) * self.p.trap_kappa * self.p.U0 / (self.k.XF_m * self.p.trap_Z0**2 * (2*pi*self.p.f_rf)**2)
        self.p.trap_mathieu_ap = 8 * (self.k.fund_e * self.k.XF_charge) * self.p.Vp / (self.k.XF_m * self.p.trap_R_eff**2 * (2*pi*self.p.f_rf)**2)
        self.p.trap_mathieu_a = abs(self.p.trap_mathieu_a0) * array([-1,-1,2]) + self.p.trap_mathieu_ap * array([1,-1,0])
        self.p.trap_mathieu_q = 2 * (self.k.fund_e * self.k.XF_charge) * self.p.V_rfpp / (self.k.XF_m * self.p.trap_R_eff**2 * (2*pi*self.p.f_rf)**2) * array([-1,1,0])
        """
            omega = 2 pi f_rf / 2 * sqrt( a + q^2 / 2 )
        """
        self.p.trap_omega = 2*pi*self.p.f_rf / 2 * sqrt(self.p.trap_mathieu_q**2 / 2 + self.p.trap_mathieu_a)
        if output_Bool == True:
            print('(fx, fy, fz): = ({:.2f}, {:.2f}, {:.2f}) kHz'.format(*self.p.trap_omega/2/pi/1e3))
        
        self.p.E_displace_ramp_time = 7/(min(self.p.trap_omega)/2/pi) # (s)
    
    # Fit slosh frequencies
    def fit_motion_freq(self, fit_params=None):
        """
            Fit slosh frequencies for all three axes and two rotations
        """
        
        self.motion_popt = [ [ [] for i in range(3) ] for Rswitch in range(2) ]
        self.motion_pcov = [ [ [] for i in range(3) ] for Rswitch in range(2) ]
        if fit_params==None:
            fit_params=[ [5e-3,each,pi/4,0] for each in self.p.trap_omega ]
        
        for j in range(2):
            for i in range(3):
                self.motion_popt[j][i],self.motion_pcov[j][i] = curve_fit(fit_sine,self.p.time_steps,self.simul_r[j][i],p0=fit_params[i])
                
        # Slosh frequencies [Hz]
        self.motion_slosh = array(self.motion_popt)[:,:,1]/2/pi
                
        self.fitted_motion = True
    
    # Plot motion of ion
    def plot_motion(self, mode=None, r_unit=None, t_unit=None, t=None, fitBool=None, plotPre=None, fit_motion_plots=None, fit_params=None, return_bool=None):
        """
            Plots motion
            Gives plots of each axis vs time separately by default
            To see 3D trajectory, use mode '3D'
        """
        
        self.motion_check()
        
        # Prefix for position
        prefix = {0:'',-3:'m',-6:'µ',-9:'n'}
        
        # Find most appropriate scale (in units of 3) based on range of positions given
        if r_unit == None:
            r_unit = int(round(log10(max(abs(self.simul_r[0].flatten())))/3)*3)
        if t_unit == None:
            t_unit = int(round(log10(max(self.p.time_steps))/3)*3)
        
        try:
            # Plot axes out separately
            if mode == None:
                
                # Fit parameters
                if fit_motion_plots!=None:
                    self.fit_motion_plots = fit_motion_plots

                if self.fit_motion_plots:
                    if not self.fitted_motion:
                        self.fit_motion_freq(fit_params=fit_params)
                
                [fig,ax] = subplots(3,2,figsize=(16,9))
                label_array = ['x','y','z']
                for j in range(2):
                    for i in range(3):
                        ax[i][j].plot(self.p.time_steps * 10**-t_unit , self.simul_r[j][i] * 10**-r_unit)
                        ax[i][j].set_xlabel('Time ('+prefix[t_unit]+'s)')
                        ax[i][j].set_ylabel(label_array[i]+' ('+prefix[r_unit]+'m)')
                        ax[i][j].grid()

                        if fitBool==None:
                            ax[i][j].plot(self.p.time_steps * 10**-t_unit , fit_sine(self.p.time_steps,*self.motion_popt[j][i]) * 10**-r_unit)
                            ax[i][j].legend(['Motion','%.2fkHz'%(self.motion_popt[j][i][1]/2000/pi)])
                
                        if plotPre==True:
                            ax[i][j].plot((self.p.kine_time_stepsblah-self.p.kine_time_stepsblah[-1]) * 10**-t_unit , self.simul_rblah[j][i] * 10**-r_unit)
                
                    ax[0][j].set_title('Rswitch = '+str(int(-2*(j-0.5))))
                fig.tight_layout(w_pad=4)
                show()
                if return_bool==True:
                    return fig
            
            # 3D plot
            elif mode =='3D':
                # Need to account for Rswitch
                fig = figure(figsize=(8,4))
                ax = [fig.add_subplot(121,projection='3d'),fig.add_subplot(122,projection='3d')]
                for n in range(2):
                    ax[n].plot(*self.simul_r[n][:,:-10]*10**-r_unit)
                    ax[n].set_xlabel('x ('+prefix[r_unit]+'m)')
                    ax[n].set_ylabel('y ('+prefix[r_unit]+'m)')
                    ax[n].set_zlabel('z ('+prefix[r_unit]+'m)')
                    ax[n].set_title('Rswitch = '+str(int(-2*(n-0.5))))
                fig.tight_layout()
                show()
                if return_bool==True:
                    return fig
            
            # Interactive 3D plot
            elif mode =='interactive':
                # Need to account for Rswitch
                def plot_int(time_step):
                    fig = figure()
                    ax = fig.add_subplot(111,projection='3d')
                    ax.plot(*self.simul_r[:,:-10]*10**-r_unit)
                    ax.scatter([(self.simul_r[:,time_step]*10**-r_unit)[0]],
                               [(self.simul_r[:,time_step]*10**-r_unit)[1]],
                               [(self.simul_r[:,time_step]*10**-r_unit)[2]],
                               s=100,color='r')
                    ax.set_xlabel('x ('+prefix[r_unit]+'m)')
                    ax.set_ylabel('y ('+prefix[r_unit]+'m)')
                    ax.set_zlabel('z ('+prefix[r_unit]+'m)')
                    show()

                interact(plot_int,time_step=(0,len(self.p.time_steps)-1))
            else:
                print('Invalid mode: '+mode+"\nAvailable modes: '3D'")
        except:
            print("Something went wrong somewhere =/")        
            
    # Solve Schrödinger's equation
    def solve_SE(self):
        """
            Solves Schrödinger's equation
            Returns expectation values of [sigmax[t], sigmay[t], sigmaz[t]] by default
            To extract ket[t], use mode 'psi'
        """
        
        self.motion_check()
        
        # Break into chunks
        if self.p.debug:
            """ In debug mode, only evaluate up to chunk number num_chunk_debug, but carry on from previous chunks, if any. """
            num_chunk = self.p.num_chunk_debug
            try:
                offset = len(self.simul_result_chunks_last_psi0[0])
                self.simul_result_chunks_asyms_cache , self.simul_result_chunks_exps_cache , self.simul_result_chunks_last_psi0_cache = self.simul_result_chunks_asyms , self.simul_result_chunks_exps , self.simul_result_chunks_last_psi0
                self.simul_result_chunks_asyms = [ [ [] for n in range(num_chunk+offset) ] for m in range(4) ] # array to store results of chunks
                self.simul_result_chunks_exps = [ [ [] for n in range(num_chunk+offset) ] for m in range(4) ] # array to store results of chunks
                self.simul_result_chunks_last_psi0 = [ [ [] for n in range(num_chunk+offset) ] for m in range(4) ] # array to store results of chunks
                for m in range(4):
                    for n in range(len(self.simul_result_chunks_last_psi0_cache[m])):
                        self.simul_result_chunks_asyms[m][n] , self.simul_result_chunks_exps[m][n] , self.simul_result_chunks_last_psi0[m][n] = self.simul_result_chunks_asyms_cache[m][n] , self.simul_result_chunks_exps_cache[m][n] , self.simul_result_chunks_last_psi0_cache[m][n]
            except:
                offset = 0
                self.simul_result_chunks_asyms = [ [ [] for n in range(num_chunk+offset) ] for m in range(4) ] # array to store results of chunks
                self.simul_result_chunks_exps = [ [ [] for n in range(num_chunk+offset) ] for m in range(4) ] # array to store results of chunks
                self.simul_result_chunks_last_psi0 = [ [ [] for n in range(num_chunk+offset) ] for m in range(4) ] # array to store results of chunks
        else:
            num_chunk = self.p.num_chunk
            offset = 0
            self.simul_result_chunks_asyms = [ [ [] for n in range(num_chunk+offset) ] for m in range(4) ] # array to store results of chunks
            self.simul_result_chunks_exps = [ [ [] for n in range(num_chunk+offset) ] for m in range(4) ] # array to store results of chunks
            self.simul_result_chunks_last_psi0 = [ [ [] for n in range(num_chunk+offset) ] for m in range(4) ] # array to store results of chunks
        total_step = len(self.p.time_steps/self.p.t_step) # number of steps in total
        total_step_per_chunk = int(total_step/self.p.num_chunk) # number of steps per chunk

        # Solve in parallel for all the blocks: BR blocks of chunks of steps of doublets [of 3D triplets]
        self.simul_result_chunks_asyms,self.simul_result_chunks_exps,self.simul_result_chunks_last_psi0 = parfor(propSolve, range(4), kwarg={'offset':offset, 'num_chunk':num_chunk, 'total_step_per_chunk':total_step_per_chunk, 'self':self})
        
        # # Dechunk into 8 blocks of list of states
        datestring = datetime.datetime.now()
        ds = datestring.strftime("%H:%M:%S")
        print('('+ds+'): Dechunking...')
        """
            1) Collapse into BR blocks of steps of doublets
            2) Flip doublet and step dimensions
            3) Break into D B R
            4) Swap B and D
            5) Collapse into BDR blocks of steps
            6) Get BDR blocks of steps; convert to float
        """
        self.simul_result_asyms = swapaxes(swapaxes(self.simul_result_chunks_asyms.reshape(4,total_step_per_chunk*(num_chunk+offset),2),1,2).reshape(2,2,2,total_step_per_chunk*(num_chunk+offset)),1,2).reshape(8,total_step_per_chunk*(num_chunk+offset)).astype(float)
        """
            1) Collapse into BR blocks of steps of doublets of 3D triplets
            2) Flip doublet and step dimensions
            3) Break into D B R blocks
            4) Swap B and D
            5) Collapse into BDR blocks of steps of 3D triplets
            6) Flip 3D triplets and step dimensions
            7) Get BDR blocks of 3D triplets of steps; convert to float
        """
        self.simul_result_exps = swapaxes(swapaxes(swapaxes(self.simul_result_chunks_exps.reshape(4,total_step_per_chunk*(num_chunk+offset),2,3),1,2).reshape(2,2,2,total_step_per_chunk*(num_chunk+offset),3),1,2).reshape(8,total_step_per_chunk*(num_chunk+offset),3),1,2).astype(float)

        datestring = datetime.datetime.now()
        ds = datestring.strftime("%H:%M:%S")
        print('('+ds+'): Done!')
        
        self.p.ss_check = True
    
    # Fit frequencies
    def fit_freq(self, fit_params=None):
        
        if fit_params==all(None):
            if len(self.p.E_rot_phase)==6:
                fit_params=[1,2*pi*25*self.p.boost,pi/4,0]
            else:
                fit_params=[ [1,2*pi*15*self.p.boost,pi/4,0] for i in range(8) ]
                
        # Prepare for plot fitting
        self.popt = [ [ [] for i in range(1+3) ] for BDR in range(len(self.p.BDR_list)) ]
        self.pcov = [ [ [] for i in range(1+3) ] for BDR in range(len(self.p.BDR_list)) ]

        # For each BDR
        for BDR in range(len(self.p.BDR_list)):

            # Fit curves
            for i in range(1+3):

                if i==0: # asymmetry
                    self.popt[BDR][0],self.pcov[BDR][0] = curve_fit(fit_sine,self.p.time_steps[:len(self.simul_result_asyms[BDR])],self.simul_result_asyms[BDR],p0=fit_params[BDR])

                else: # Pauli
                    self.popt[BDR][i],self.pcov[BDR][i] = curve_fit(fit_sine,self.p.time_steps[:len(self.simul_result_exps[BDR][i-1])],self.simul_result_exps[BDR][i-1].astype(float),p0=fit_params[BDR])
                
        self.fitted = True
        self.fs = array(self.popt)[:,0,1]/2/pi # frequency fits to asymmetry
    
    # Plot state
    def plot_state(self, mode=None, fit_plots=None, fit_params=None, return_bool=None):
        
        self.state_check()
        
        # Fit parameters
        if fit_plots!=None:
            self.fit_plots = fit_plots
            
        if self.fit_plots:
            if not self.fitted:
                self.fit_freq(fit_params=fit_params)
        
        # Plot against time
        if mode == None:

            # Same BRD across rows, columns are [asymmetry, sigmax, sigmay, sigmaz]
            [fig,ax] = subplots(len(self.p.BDR_list),1+3+1,figsize=(20,3*len(self.p.BDR_list)))
            label_array = ['Asymmetry',r'$\langle \sigma_x \rangle$',r'$\langle \sigma_y \rangle$',r'$\langle \sigma_z \rangle$','Bloch norm']
            for BDR in range(len(self.p.BDR_list)): # for each row

                for i in range(1+3+1): # for each column

                    # Split into Asymmetry plot and Pauli matrices plots
                    if i==0: # asymmetry
                        ax[BDR][0].plot(self.p.time_steps[:len(self.simul_result_asyms[BDR])],self.simul_result_asyms[BDR])

                        if self.fit_plots:
                            ax[BDR][0].plot(self.p.time_steps[:len(self.simul_result_asyms[BDR])],fit_sine(self.p.time_steps[:len(self.simul_result_asyms[BDR])],*(self.popt[BDR][0])))
                            ax[BDR][0].legend(['%.3gHz'%abs(self.popt[BDR][0][1]/2/pi)])
                    elif i==4: # Bloch norm
                     ax[BDR][4].plot(self.p.time_steps[:len(self.simul_result_exps[BDR][0])],sqrt(self.simul_result_exps[BDR][0]**2+self.simul_result_exps[BDR][1]**2+self.simul_result_exps[BDR][2]**2))
                    else: # Pauli
                        ax[BDR][i].plot(self.p.time_steps[:len(self.simul_result_exps[BDR][i-1])],self.simul_result_exps[BDR][i-1])

                        if self.fit_plots:    
                            ax[BDR][i].plot(self.p.time_steps[:len(self.simul_result_exps[BDR][i-1])],fit_sine(self.p.time_steps[:len(self.simul_result_exps[BDR][i-1])],*(self.popt[BDR][i])))
                            ax[BDR][i].legend(['%.3gHz'%abs(self.popt[BDR][i][1]/2/pi)])
                    ax[BDR][i].set_xlabel('Time (s)')
                    ax[BDR][i].set_ylabel(label_array[i])
                    ax[BDR][i].set_title(self.p.BDR_list[BDR])
                    ax[BDR][i].grid()
            fig.tight_layout(w_pad=0)
            if return_bool==True:
                    return fig

        # 3D plots on Bloch spheres
        elif mode == '3D':
            fig = figure(figsize=(12,int(3*len(self.p.BDR_list)/4)))
            for BDR in range(len(self.p.BDR_list)):
                ax = fig.add_subplot(int(len(self.p.BDR_list)/4),4,BDR+1, projection='3d')
                b = Bloch(fig=fig, axes=ax)
                b.add_points(self.simul_result_exps[BDR],meth='l') # history line
                b.add_vectors([each[-1] for each in self.simul_result_exps[BDR]]) # current position
                b.font_size = 12
                b.render(fig=fig, axes=ax)
                ax.set_aspect('equal')
                ax.set_title(self.p.BDR_list[BDR],fontdict={'fontsize':16},loc='left')
            fig.tight_layout()
            show()
            if return_bool==True:
                    return fig
        elif mode == 'population':
            fig,ax = subplots(2,4,figsize=(16,6))

            q_num_labels = [ r'$|\Omega={:d}, F={:.1f}, m_F={:.1f}\rangle$'.format(int(each[1]),each[2],each[3]) for each in self.q.q_nums ]

            dx,dy = 0.2,0.005

            for BR in range(4):
                for doub in range(2):
                    for st in range(12):
                        ax[int(floor(BR/2))][BR%2 + 2*doub].add_artist( Rectangle( ( self.q.q_nums[st,3] - dx/2, 
                                                                                     self.q.q_nums[st,2]*-1 - self.q.q_nums[st,1]*self.q.q_nums[st,3]*0.2 - dy/2
                                                                                   ),
                                                                                   dx,
                                                                                   dy,
                                                                                   color='tab:grey',fill=False
                                                                             )
                                                                      )
                        ax[int(floor(BR/2))][BR%2 + 2*doub].add_artist( Circle( (self.q.q_nums[st,3], 
                                                                                 self.q.q_nums[st,2]*-1 - self.q.q_nums[st,1]*self.q.q_nums[st,3]*0.2
                                                                                ), 
                                                                               radius=0.4*(abs(self.simul_result_chunks_last_psi0[BR,-1,doub].full())**2).flatten()[st]
                                                                              )
                                                                      )
            #         ax[int(floor(BR/2))][BR%2 + 2*doub].barh([ i for i in range(12) ], (abs(self.simul_result_chunks_last_psi0[BR,-1,doub].full())**2).flatten(), tick_label=q_num_labels)
                    ax[int(floor(BR/2))][BR%2 + 2*doub].set_title('[{0}, {2}, {1}]'.format(*self.p.BR_list[BR],int((doub-0.5)*-2)))
                    ax[int(floor(BR/2))][BR%2 + 2*doub].set_ylim([-2.3,0.2])
                    ax[int(floor(BR/2))][BR%2 + 2*doub].set_xlim([-2,2])
                    ax[int(floor(BR/2))][BR%2 + 2*doub].set_xticks([])
                    ax[int(floor(BR/2))][BR%2 + 2*doub].set_yticks([])
            fig.tight_layout()
            show()
            if return_bool==True:
                    return fig
        else:
            print('Invalid mode '+mode+"\nAvailable modes: 'None', '3D'")

    # Plot norm of Bloch vector
    def plot_norm(self, return_bool=None):
        """
            Plots (sigmax^2 + sigmay^2 + sigmaz^2) against time
        """
        
        self.state_check()
        
        try:
            [fig,ax] = subplots(len(self.p.BDR_list),1,figsize=(6,2*len(self.p.BDR_list)))
            for BDR in range(len(self.p.BDR_list)):
                ax[BDR].plot(self.p.time_steps[:len(self.simul_result_exps[BDR][0])],self.simul_result_exps[BDR][0]**2+self.simul_result_exps[BDR][1]**2+self.simul_result_exps[BDR][2]**2)
                ax[BDR].set_xlabel('Time (s)')
                ax[BDR].set_ylabel('Bloch norm')
                ax[BDR].set_title(self.p.BDR_list[BDR])
            fig.tight_layout(w_pad=4)
            if return_bool==True:
                    return fig
        except:
            print("Something went wrong somewhere =/")
            
    # Calculate f^X's
    def get_fs(self):
        """
            Calculates and prints f^{}. Also gives error in fitting
        """
        try:
            BDR_list = array(self.p.BDR_list)
            fs = array([ [ abs(each[1]/2/pi) for each in eacheach] for eacheach in self.popt ]).T
            self.f0 = [mean(f) for f in fs]
            print('- f0: {0:.3g}, {1:.3g}, {2:.3g}, {3:.3g}'.format(*self.f0))
            self.fB = [mean(f*BDR_list[:,0]) for f in fs]
            print('- fB: {0:.3g}, {1:.3g}, {2:.3g}, {3:.3g}'.format(*self.fB))
            self.fD = [mean(f*BDR_list[:,1]) for f in fs]
            print('- fD: {0:.3g}, {1:.3g}, {2:.3g}, {3:.3g}'.format(*self.fD))
            self.fR = [mean(f*BDR_list[:,2]) for f in fs]
            print('- fR: {0:.3g}, {1:.3g}, {2:.3g}, {3:.3g}'.format(*self.fR))
            self.fBD = [mean(f*BDR_list[:,0]*BDR_list[:,1]) for f in fs]
            print('- fBD: {0:.3g}, {1:.3g}, {2:.3g}, {3:.3g}'.format(*self.fBD))
            self.fBR = [mean(f*BDR_list[:,0]*BDR_list[:,2]) for f in fs]
            print('- fBR: {0:.3g}, {1:.3g}, {2:.3g}, {3:.3g}'.format(*self.fBR))
            self.fDR = [mean(f*BDR_list[:,1]*BDR_list[:,2]) for f in fs]
            print('- fDR: {0:.3g}, {1:.3g}, {2:.3g}, {3:.3g}'.format(*self.fDR))
            self.fBDR = [mean(f*BDR_list[:,0]*BDR_list[:,1]*BDR_list[:,2]) for f in fs]
            print('- fBDR: {0:.3g}, {1:.3g}, {2:.3g}, {3:.3g}'.format(*self.fBDR))

            # Error from asymmetry and Pauli's
            self.fErrAll = sqrt(sum([ [diag(each)[1]/2/pi for each in eacheach] for eacheach in self.pcov ]))/8 # ∆f = [ ∑ (∂f)^2 ]/8
            print('- ∆f_all = {0:.2g}'.format(self.fErrAll))

            # Error from asymmetry only
            self.fErr = sqrt(sum([ diag(eacheach[0])[1]/2/pi for eacheach in self.pcov ]))/8 # ∆f = [ ∑ (∂f)^2 ]/8
            print('- ∆f = {0:.2g}'.format(self.fErr))
        except:
            print('Error in calculating frequencies.')

    # Hamiltonian for debugging and quick diagonalization
    def H(self, E=None, B=None, alpha=None, B_nr=None, alpha_nr=None, Bswitch=None, Rswitch=None):
        """
            E,B: array
            alpha, B_nr, alpha_nr: float
            Bswitch, Rswitch: ±1
        
            By default, calculate the total Hamiltonian for:
            - If E/B not given, assuming E=array([0,0,Erot]), B=array([0,0,Brot])
            - Perfect Erot, Baxgrad; not sampling trap inhomogeneity
            - Non-reversing Baxgrad included through self.p.B_axial_nonreversing unless B_nr is given, also including B0_nr
            - Bswitch=Rswitch=+1 by default
        """
        if Bswitch==None:
            Bswitch=1
        if Rswitch==None:
            Rswitch=1
        
        # Perfect Erot
        if all(E)==None:
            E = array([0,0,self.p.E_rot])
        
        # Perfect Baxgrad
        if all(B)==None:
            B=array([0,0,self.k.fund_e*self.p.E_rot / (self.k.XF_m * (2*pi*self.p.f_rot)**2) * self.p.B_axial_grad_reversing])
        if B_nr==None:
            Bnr = array([0,0,self.k.fund_e*self.p.E_rot / (self.k.XF_m * (2*pi*self.p.f_rot)**2) * self.p.B_axial_grad_nonreversing])
        else:
            Bnr = array([0,0,B_nr])
        
        # Plan B
        if alpha==None:
            alpha = 0
        if alpha_nr==None:
            alpha_nr = 0
        
        return Qobj((self.q.H_hf 
                     + self.q.H_Omega_doubling 
                     + self.q.H_Rot(Rswitch,kk=self.k,pp=self.p)
                     + self.p.V_dit_bool * self.q.mF * (alpha*Rswitch + alpha_nr) * self.p.f_rot # Adding in by hand for Plan B for diagonalization, logic is the same behind how we model Brot
                     + self.q.H_Stark(E,kk=self.k,pp=self.p)
                     + self.q.H_Zeeman((B*Bswitch + Bnr)+to_rot(self.p.B0_nr,pi/2,0),kk=self.k,pp=self.p)
                     + self.q.offset))
            
    # Estimate f^X's
    def est_fs(self, E=None, B=None, alpha=None, B_nr=None, alpha_nr=None, output_Bool=None, diag_or_expand_Bool=None, planB_Bool=None):
        """
            output_Bool: whether to show calculated fs
            diag_or_expand_Bool: True for expand, whether to diagonalize Hamiltonian (default) or use small variable expansion expressions
            planB_Bool: True for Plan B, Plan A otherwise
            
            Check documentation for self.H().
        """
        
        if all(E)==None:
            E = array([0,0,self.p.E_rot])
        if all(B)==None:
            B=array([0,0,self.k.fund_e*self.p.E_rot / (self.k.XF_m * (2*pi*self.p.f_rot)**2) * self.p.B_axial_grad_reversing])
        if B_nr==None:
            Bnr = array([0,0,self.k.fund_e*self.p.E_rot / (self.k.XF_m * (2*pi*self.p.f_rot)**2) * self.p.B_axial_grad_nonreversing])
        else:
            Bnr = array([0,0,B_nr])
        
        # Plan B
        if alpha==None:
            alpha = self.p.E_rot_a / 2 * self.p.E_dit/self.p.E_rot
        if alpha_nr==None:
            alpha_nr = 0
        
        ### Obtain fs
        if diag_or_expand_Bool==None: # direct diagonalization
            
            if E[2] < (11.4e2 if self.k.ThF_bool else 34.6e2):
                shiftIndex = -2
            else:
                shiftIndex = 0
                
            # Find eigenvalues for the upper(-3,-4) and lower doublets(1,0) for BR with R varying quickest
            est_Hs = array([ sort(self.H(E=E,B=B,alpha=alpha,B_nr=B_nr,alpha_nr=alpha_nr,Bswitch=Bswitch,Rswitch=Rswitch).eigenenergies())[[9+shiftIndex,8+shiftIndex,1,0]] for Bswitch in [1,-1] for Rswitch in [1,-1] ])
            
            # Rearrange into BDR with R varying quickest
            est_fs = swapaxes(array([est_Hs[:,0]-est_Hs[:,1],est_Hs[:,-2]-est_Hs[:,-1]]).reshape((2,2,2)),0,1).flatten().real
            
            # Calculate
            BDR_list = array(self.p.BDR_list)
            self.est_f0 = mean(est_fs)
            self.est_fB = mean(est_fs*BDR_list[:,0])
            self.est_fD = mean(est_fs*BDR_list[:,1])
            self.est_fR = mean(est_fs*BDR_list[:,2])
            self.est_fBD = mean(est_fs*BDR_list[:,0]*BDR_list[:,1])
            self.est_fBR = mean(est_fs*BDR_list[:,0]*BDR_list[:,2])
            self.est_fDR = mean(est_fs*BDR_list[:,1]*BDR_list[:,2])
            self.est_fBDR = mean(est_fs*BDR_list[:,0]*BDR_list[:,1]*BDR_list[:,2])
            self.est_fall = array([self.est_f0,self.est_fB,self.est_fD,self.est_fR,self.est_fBD,self.est_fBR,self.est_fDR,self.est_fBDR])
            self.est_fraw = est_fs
            
        else: # expansion, all EDM components set to zero
            if self.ested==False:
                self.est_para()
            if self.ested_Para==False:
                self.est_para_Berry()
        
            if planB_Bool==None:
        
                self.est_f0 = 3 * self.g * self.k.fund_uB * B[2]/self.k.fund_hbar/2/pi * ( 1 + (self.Delta**2 + self.dDelta**2)/2/(3 * self.g * self.k.fund_uB * B[2]/self.k.fund_hbar/2/pi)**2 )
                self.est_fB = ( 3 * self.g * self.k.fund_uB * Bnr[2]/self.k.fund_hbar/2/pi + 3 * self.p.f_rot * alpha_nr ) * ( 1 - (self.Delta**2 + self.dDelta**2)/2/(3 * self.g * self.k.fund_uB * B[2]/self.k.fund_hbar/2/pi)**2 )
                self.est_fD = 3 * self.dg * self.k.fund_uB * B[2]/self.k.fund_hbar/2/pi + self.Delta * self.dDelta / ( 3 * self.g * self.k.fund_uB * B[2]/self.k.fund_hbar/2/pi )
                self.est_fR = 3 * self.p.f_rot * alpha / (3 * self.g * self.k.fund_uB * B[2]/self.k.fund_hbar/2/pi)**3 * ( (self.Delta**2 + self.dDelta**2) * ( 3 * self.g * self.k.fund_uB * Bnr[2]/self.k.fund_hbar/2/pi + 3 * self.p.f_rot * alpha_nr ) )
                self.est_fBD = 3 * self.dg * self.k.fund_uB * Bnr[2]/self.k.fund_hbar/2/pi - self.Delta*self.dDelta * ( 3 * self.g * self.k.fund_uB * Bnr[2]/self.k.fund_hbar/2/pi + 3 * self.p.f_rot * alpha_nr ) / (3 * self.g * self.k.fund_uB * B[2]/self.k.fund_hbar/2/pi)**2
                self.est_fBR = 3 * self.p.f_rot * alpha * ( 1 -(self.Delta**2 + self.dDelta**2)/2/(3 * self.g * self.k.fund_uB * B[2]/self.k.fund_hbar/2/pi)**2 )
                self.est_fDR = self.est_fR * 2*self.Delta*self.dDelta/(self.Delta**2 + self.dDelta**2)
                self.est_fBDR = 3 * self.p.f_rot * alpha / (3 * self.g * self.k.fund_uB * B[2]/self.k.fund_hbar/2/pi)**2 * (-self.Delta * self.dDelta + self.dg/self.g*(self.Delta**2 + self.dDelta**2))
                
            else:
                
                self.est_f0 = self.DmF * self.p.f_rot * alpha * ( 1 + (self.Delta**2 + self.dDelta**2)/2/(self.DmF * self.p.f_rot * alpha)**2 )
                self.est_fD = self.dDmF * self.p.f_rot * alpha + self.Delta*self.dDelta/(self.DmF * self.p.f_rot * alpha) * (1 - (self.Delta**2 + self.dDelta**2)/2/(self.DmF * self.p.f_rot * alpha)**2 )
                self.est_fR = ( 3 * self.g * self.k.fund_uB * Bnr[2]/self.k.fund_hbar/2/pi + self.DmF * self.p.f_rot * alpha_nr ) * ( 1 - (self.Delta**2 + self.dDelta**2)/2/(self.DmF * self.p.f_rot * alpha)**2 )
                self.est_fDR = 3 * self.dg * self.k.fund_uB * Bnr[2]/self.k.fund_hbar/2/pi + self.dDmF * self.p.f_rot * alpha_nr - self.Delta*self.dDelta * ( 3 * self.g * self.k.fund_uB * Bnr[2]/self.k.fund_hbar/2/pi + self.DmF * self.p.f_rot * alpha_nr ) / (self.DmF * self.p.f_rot * alpha)**2
        
        if output_Bool==True:
        
            if planB_Bool==None:
                print("Estimated f^X's for Plan A")
                print('- f0: {0:.3g}'.format(self.est_f0))
                print('- fB: {0:.3g}'.format(self.est_fB))
                print('- fD: {0:.3g}'.format(self.est_fD))
                print('- fR: {0:.3g}'.format(self.est_fR))
                print('- fBD: {0:.3g}'.format(self.est_fBD))
                print('- fBR: {0:.3g}'.format(self.est_fBR))
                print('- fDR: {0:.3g}'.format(self.est_fDR))
                print('- fBDR: {0:.3g}'.format(self.est_fBDR))
            else:
                print("Estimated f^X's for Plan B")
                print('- f0: {0:.3g}'.format(self.est_f0))
                print('- fD: {0:.3g}'.format(self.est_fD))
                print('- fS: {0:.3g}'.format(self.est_fR))
                print('- fDS: {0:.3g}'.format(self.est_fDR))

    # Calculate ∆, d∆, g, dg, g_eff_grad, dg_eff_grad, g_eff_quo, dg_eff_quo
    def est_para(self,output_Bool=None,B_max=None,Brot=None,return_fit=None):
        """
            Calculates ∆, d∆, g, dg, g_eff_grad, dg_eff_grad, g_eff_quo, dg_eff_quo for a given set of parameters defined in self.p.
            
            #_grad refers to d/dBrot·f0
                - To find the gradient of the avoided crossing
                
            #_quo refers to f0·/Brot
                - To find the equivalent g that gives the f0 splitting
                
                At large Brot, #_grad = #_quo
        """
        
        # Save estimated fXs
        try:
            cache_f0,cache_fB,cache_fD,cache_fR,cache_fBD,cache_fBR,cache_fDR,cache_fBDR = self.est_f0,self.est_fB,self.est_fD,self.est_fR,self.est_fBD,self.est_fBR,self.est_fDR,self.est_fBDR
            cache_bool = True
        except AttributeError:
            cache_bool = False

        
        # Prepare data containers
        if B_max==None:
            B_max = 1e-3 * 1e-4 # [T]
        Bs = linspace(-B_max, B_max, num=2*10+1)
        Bs_fit = linspace(-B_max, B_max, num=2*100+1)
        self.fss = []
        
        if Brot==None:
            Brot = 0.798e-3 * 1e-4 # [T]

        # Calculate f^u and f^l
        for B in Bs:
            self.est_fs(B=array([0,0,B]))
            self.fss.append([self.est_f0+self.est_fD , self.est_f0-self.est_fD])

        # Data manipulation
        self.fss = array(self.fss).real

        # ∆, d∆
        self.Delta_u , self.Delta_l = self.fss[int(len(Bs)/2),0] , self.fss[int(len(Bs)/2),1]
        self.Delta , self.dDelta = (self.Delta_u+self.Delta_l)/2 , (self.Delta_u-self.Delta_l)/2
        
        # Fitting function for g and dg
        def fit_avoided_crossing(B,g,Delta):
            """ Fit to avoided crossing curve to extract g and dg """
            return sqrt( (3 * g * self.k.fund_uB * B / self.k.fund_hbar /2/pi)**2 + Delta**2 )

        # Fit for g and dg
        poptu,pcovu = curve_fit(fit_avoided_crossing,Bs,self.fss[:,0],p0=[0.001,self.Delta_u])
        poptl,pcovl = curve_fit(fit_avoided_crossing,Bs,self.fss[:,1],p0=[0.001,self.Delta_l])

        # g , dg
        self.g_u , self.g_l = poptu[0] , poptl[0]
        self.g , self.dg = (self.g_u+self.g_l)/2 , (self.g_u-self.g_l)/2
        
        self.ested = True
        
        # g_eff_grad , dg_eff_grad
        g_eff_grads = [ gradient(each , Bs) * self.k.fund_hbar * 2*pi / self.k.fund_uB / 3 for each in [ self.fss[:,0] , self.fss[:,1] ] ]
        [ self.g_eff_grad_u , self.g_eff_grad_l ] = [ interp(Brot,Bs,each) for each in g_eff_grads ]
        self.g_eff_grad , self.dg_eff_grad = (self.g_eff_grad_u+self.g_eff_grad_l)/2 , (self.g_eff_grad_u-self.g_eff_grad_l)/2
        
        # g_eff_quo , dg_eff_quo
        [ self.g_eff_quo_u , self.g_eff_quo_l ] = [ interp(Brot,Bs,each) * self.k.fund_hbar * 2*pi / (3 * self.k.fund_uB * Brot)  for each in [ self.fss[:,0] , self.fss[:,1] ] ]
        self.g_eff_quo , self.dg_eff_quo = (self.g_eff_quo_u+self.g_eff_quo_l)/2 , (self.g_eff_quo_u-self.g_eff_quo_l)/2
        
        if output_Bool==True:
            figure(figsize=(6,4))
            plot(Bs*1e4*1e3,self.fss[:,0],'^')
            plot(Bs*1e4*1e3,self.fss[:,1],'v')
            plot(Bs_fit*1e4*1e3,fit_avoided_crossing(Bs_fit,*poptu))
            plot(Bs_fit*1e4*1e3,fit_avoided_crossing(Bs_fit,*poptl))
            grid()
            xlabel('$B_\mathrm{rot}$ (mG)')
            ylabel('$f^0$ (Hz)')
            legend(['Upper','Lower'])
            tight_layout()
            show()
            
            print('Diagonalized {}-level Hamiltonian\n'.format(len(self.q.q_nums)))
            
            print('- ∆: {0:.3g}'.format(self.Delta))
            print('- d∆: {0:.3g}'.format(self.dDelta))
            print('- ∆_u: {0:.4g}'.format(self.Delta_u))
            print('- ∆_l: {0:.4g}\n'.format(self.Delta_l))

            print('- g: {0:.4g}'.format(self.g))
            print('- dg: {0:.3g}'.format(self.dg))
            print('- dg/g: {0:.3g}\n'.format(self.dg/self.g))

            print('- g_eff_grad: {0:.4g}'.format(self.g_eff_grad))
            print('- dg_eff_grad: {0:.3g}'.format(self.dg_eff_grad))
            print('- dg_eff_grad/g_eff_grad: {0:.3g}\n'.format(self.dg_eff_grad/self.g_eff_grad))

            print('- g_eff_quo: {0:.4g}'.format(self.g_eff_quo))
            print('- dg_eff_quo: {0:.3g}'.format(self.dg_eff_quo))
            print('- dg_eff_quo/g_eff_quo: {0:.3g}'.format(self.dg_eff_quo/self.g_eff_quo))
        
        if cache_bool:
            self.est_f0,self.est_fB,self.est_fD,self.est_fR,self.est_fBD,self.est_fBR,self.est_fDR,self.est_fBDR = cache_f0,cache_fB,cache_fD,cache_fR,cache_fBD,cache_fBR,cache_fDR,cache_fBDR
            del cache_f0,cache_fB,cache_fD,cache_fR,cache_fBD,cache_fBR,cache_fDR,cache_fBDR
        
        if return_fit==True:
            return poptu,pcovu,poptl,pcovl
              
        del Bs,g_eff_grads,cache_bool,poptu,pcovu,poptl,pcovl

    # Calculate ∆, d∆, ∆mF, d∆mF
    def est_para_Berry(self,output_Bool=None,alpha_max=None,alpha=None):
        """
            Calculates ∆, d∆, ∆mF, d∆mF, for a given set of parameters defined in self.p.
            
            #_grad refers to d/dBrot·f0
            - To find the gradient of the avoided crossing
            
            #_quo refers to f0·/Brot
            - To find the equivalent g that gives the f0 splitting
            
            At large Brot, #_grad = #_quo
            """
        
        # Save estimated fXs
        try:
            cache_f0,cache_fB,cache_fD,cache_fR,cache_fBD,cache_fBR,cache_fDR,cache_fBDR = self.est_f0,self.est_fB,self.est_fD,self.est_fR,self.est_fBD,self.est_fBR,self.est_fDR,self.est_fBDR
            cache_bool = True
        except AttributeError:
            cache_bool = False
    
        if alpha==None:
            alpha0 = 0.001/4.5 # for f0 = 100 Hz when f_rot = 150 kHz
        if alpha_max==None:
            alpha_max = 1.1*alpha0
    
        # Prepare data containers
        alphas = linspace(-alpha_max,alpha_max,num=2*10+1)
        alphas_fit = linspace(-alpha_max,alpha_max,num=2*100+1)
        self.fss = []
        
        # Calculate f^u and f^l
        for alpha in alphas:
            self.est_fs(B=array([0,0,0]),alpha=alpha)
            self.fss.append([self.est_f0+self.est_fD , self.est_f0-self.est_fD])

        # Data manipulation
        self.fss = array(self.fss).real
        
        # ∆, d∆
        self.Delta_u , self.Delta_l = self.fss[int(len(alphas)/2),0] , self.fss[int(len(alphas)/2),1]
        self.Delta , self.dDelta = (self.Delta_u+self.Delta_l)/2 , (self.Delta_u-self.Delta_l)/2
    
        # Fitting function for g and dg
        def fit_avoided_crossing(alpha,DmF,Delta):
            """ Fit to avoided crossing curve to extract g and dg """
            return sqrt( (DmF * self.p.f_rot * alpha)**2 + Delta**2 )
        
        # Fit for g and dg
        poptu,pcovu = curve_fit(fit_avoided_crossing,alphas,self.fss[:,0],p0=[3,self.Delta_u])
        poptl,pcovl = curve_fit(fit_avoided_crossing,alphas,self.fss[:,1],p0=[3,self.Delta_l])
        
        # ∆mF , d∆mF
        self.DmF_u , self.DmF_l = poptu[0] , poptl[0]
        self.DmF , self.dDmF = (self.DmF_u+self.DmF_l)/2 , (self.DmF_u-self.DmF_l)/2
        
        self.ested_Para = True

        if output_Bool==True:
            figure(figsize=(6,4))
            plot(alphas*self.p.f_rot,self.fss[:,0],'^')
            plot(alphas*self.p.f_rot,self.fss[:,1],'v')
            plot(alphas_fit*self.p.f_rot,fit_avoided_crossing(alphas_fit,*poptu))
            plot(alphas_fit*self.p.f_rot,fit_avoided_crossing(alphas_fit,*poptl))
            grid()
            xlabel(r'$f_\mathrm{rot} \langle \alpha \rangle$ (Hz)')
            ylabel('$f^0$ (Hz)')
            legend(['Upper','Lower'])
            tight_layout()
            show()
            
            print('Diagonalized {}-level Hamiltonian\n'.format(len(self.q.q_nums)))
            
            print('- ∆: {0:.3g}'.format(self.Delta))
            print('- d∆: {0:.3g}'.format(self.dDelta))
            print('- ∆_u: {0:.4g}'.format(self.Delta_u))
            print('- ∆_l: {0:.4g}\n'.format(self.Delta_l))
            
            print('- ∆mF: {0:.6g}'.format(self.DmF))
            print('- d∆mF: {0:.3g}'.format(self.dDmF))
            print('- d∆mF/∆mF: {0:.3g}\n'.format(self.dDmF/self.DmF))

        if cache_bool:
            self.est_f0,self.est_fB,self.est_fD,self.est_fR,self.est_fBD,self.est_fBR,self.est_fDR,self.est_fBDR = cache_f0,cache_fB,cache_fD,cache_fR,cache_fBD,cache_fBR,cache_fDR,cache_fBDR
            del cache_f0,cache_fB,cache_fD,cache_fR,cache_fBD,cache_fBR,cache_fDR,cache_fBDR

        del alphas,alphas_fit,poptu,pcovu,poptl,pcovl,cache_bool

    # Show energy states
    def show_states(self,E=None,B=None,threshold=None,return_bool=None):
        """ Takes in E [V/m] (can be array) and plots energy levels """
       
        if all(E)==None:
            E = array([0,0,self.p.E_rot])
        elif type(E) is not ndarray:
            E = array([0,0,E])
        if all(B)==None:
            B = array([0,0,0])
        elif type(B) is not ndarray:
            B = array([0,0,B])
        if threshold==None:
            threshold=0.5
        
        if len(self.q.q_nums)==12: # J=1 only
            StarkEig_QT(self,E,B=B)
        
            ### Visualize states

            # Data manipulation

            J1 = self.q.eigenval_QT

            """ Eigenstates are linear combinations of stretched states. But they correspond to the same Ω·mF, so use this as a marker. """
            mF_coord = array([ mean(prod(self.q.q_nums[(abs(each.full()) > threshold).flatten()][:,[1,3]],axis=1)) for each in self.q.eigenvec_QT ])

            F_coord = array([ mean(self.q.q_nums[(abs(each.full()) > threshold).flatten()][:,2]) for each in self.q.eigenvec_QT ])

            # Plot

            fig,ax = subplots(1,1,figsize=(12,6))

            # J=1
            for n in range(12):
                ax.annotate(r'{2}: $|F={1:.1f},\Omega = {0:d}\rangle$'.format((-1)**n,F_coord[n],n),(mF_coord[n]*(-1)**n,J1[n]/1e6),fontsize=10,ha='center')

            ax.set_ylim([min(J1/1e6)-5,max(J1/1e6)+5])
            ax.set_xlim([-3,3])

            ax.set_ylabel('Energy (MHz)')
            ax.set_xlabel('$m_F$')
            ax.set_xticks([-2.5,-1.5,-0.5,0.5,1.5,2.5])
            ax.set_title('{}$^+$'.format('ThF' if self.k.ThF_bool else 'HfF')+': X $^3\Delta_1(v=0,J=1)$ , $\mathcal{E}_\mathrm{rot} = '+'{0:.3g}'.format(E[2]/1e2)+' \mathrm{V/cm}$')
            ax.grid()

            fig.tight_layout()
            show()
        
        else: # J=1&2
            StarkEig_QT(self,E,B=B)
        
            ### Visualize states

            # Data manipulation

            J1 = self.q.eigenval_QT[:12]
            J2 = self.q.eigenval_QT[12:]

            """ Eigenstates are linear combinations of stretched states. But they correspond to the same Ω·mF, so use this as a marker. """
            mF_coord = array([ mean(prod(self.q.q_nums[(abs(each.full()) > threshold).flatten()][:,[1,3]],axis=1)) for each in self.q.eigenvec_QT ])
            mF_coord_J1 = mF_coord[:12]
            mF_coord_J2 = mF_coord[12:]

            F_coord = array([ mean(self.q.q_nums[(abs(each.full()) > threshold).flatten()][:,2]) for each in self.q.eigenvec_QT ])
            F_coord_J1 = F_coord[:12]
            F_coord_J2 = F_coord[12:]

            # Plot

            fig,ax = subplots(2,1,figsize=(12,12))

            # J=1
            for n in range(12):
                ax[1].annotate(r'{2}: $|F={1:.1f},\Omega = {0:d}\rangle$'.format((-1)**n,F_coord_J1[n],n),(mF_coord_J1[n]*(-1)**n,J1[n]/1e6),fontsize=10,ha='center')

            ax[1].set_ylim([min(J1/1e6)-5,max(J1/1e6)+5])
            ax[1].set_xlim([-3,3])

            ax[1].set_ylabel('Energy (MHz)')
            ax[1].set_xlabel('$m_F$')
            ax[1].set_xticks([-2.5,-1.5,-0.5,0.5,1.5,2.5])
            ax[1].set_title('$J=1 , \mathcal{E}_\mathrm{rot} = '+'{:.3g}'.format(E[2]/1e2)+' \mathrm{V/cm}$')
            ax[1].grid()

            # J=2
            for n in range(len(J2)):
                ax[0].annotate(r'{2}: $|F={1:.1f},\Omega = {0:d}\rangle$'.format((-1)**n,F_coord_J2[n],n+12),(mF_coord_J2[n]*(-1)**n,J2[n]/1e6),fontsize=10,ha='center')

            ax[0].set_ylim([min(J2/1e6)-5,max(J2/1e6)+5])
            ax[0].set_xlim([-3,3])

            ax[0].set_ylabel('Energy (MHz)')
            ax[0].set_xlabel('$m_F$')
            ax[0].set_xticks([-2.5,-1.5,-0.5,0.5,1.5,2.5])
            ax[0].set_title('$J=2 , \mathcal{E}_\mathrm{rot} = '+'{:.3g}'.format(E[2]/1e2)+' \mathrm{V/cm}$')
            ax[0].grid()

            fig.tight_layout()
            show()
            
        if return_bool==True:
            return fig

    # Copy handle's motion to self
    def copy_motion(self,handle):
        """
            Copy handle's motion output to self
        """
        self.p.ts_check = handle.p.ts_check
        self.simul_v_p = handle.simul_v_p
        self.simul_r_p = handle.simul_r_p
        self.simul_v_m = handle.simul_v_m
        self.simul_r_m = handle.simul_r_m
        self.simul_v = handle.simul_v
        self.simul_r = handle.simul_r
        
    # Save object
    def save(self,fn=None,prefix=None,postfix=None):
        
        if fn==None:
            """
                Easy reference: https://www.w3schools.com/python/python_datetime.asp
                
                For 27 January 2019, 14:51:44, return: 27Jan2019_145144.obj
            """
            datestring = datetime.datetime.now()
            fn = datestring.strftime("%d%b%Y_%H%M%S")
            
        if prefix!=None:
            fn = prefix+'_'+fn
        
        # Remove filetype extension if any
        if fn[-4:]=='.obj':
            fn = fn[:-4]
        
        if postfix!=None:
            fn = fn+'_'+postfix

        # Add filetype extension
        fn = fn+'.obj'

        print('Saving state...')
        with gzip.open(fn, 'wb') as f_handle:
            pickle.dump(self, f_handle)
        print('Saved as `'+fn+'`!')


####------------ Helper Classes ------------####

class k:
    'Defining constants'
    def __init__(self):
        
        # Fundamental
        self.fund_c = 299792458 # speed of light [m/s]
        self.fund_kB = 1.3806488e-23 # Boltzmann constant [J/K]
        self.fund_epsilon0 = 8.845e-12 # permittivity [F/m]
        self.fund_mu0 = 4*pi*1e-7 # permeability [gauss m/A]
        self.fund_hbar = 1.05457173e-34 # hbar [J s]
        self.fund_e = 1.60217657e-19 # electron charge [C]
        self.fund_a0 = 5.29177210e-11 # Bohr radius [m]
        self.fund_uB = 9.274009994e-24 # Bohr magneton [J/T]
        self.fund_uN = 5.05050783888e-27 # nuclear magneton [J/T]
        self.fund_in = 0.0254 # inch in metres [m]
        
        # Conversion
        self.conv_Debye_to_SI = 3.33564e-30 # [C m / D]
        
        # ThF
        self.XF_m = 251 * 1.66053892e-27 # [kg]
        self.XF_charge = 1 # [of e]
        self.XF_A_par = -20.1 * 1e6 # hyperfine A_parallel -20.1(1)MHz [Hz]
        self.XF_omega_ef = 5.29 * 1e6 # Ω doubling 5.29(5)MHz [Hz]
        self.XF_d_mf = 3.37 # molecular frame dipole moment 3.37(4)D [D], 1.696MHz/(V/cm); measured 28.6.2019
        self.XF_Lambda = 2 # 3D1
        self.XF_gN = 5.25774 # [unitless] that of F-19 (N. J. Stone, At. Data Nucl. Data Tables 90, 75 (2005))
        self.XF_gF = -0.0149 # ThF+ -0.0149(3) [unitless]; measured 6,7.2.2020
        self.XF_Gpar = -3*self.XF_gF + self.XF_gN * self.fund_uN/self.fund_uB # [unitless] Only holds for 3∆1
        self.XF_Joffset = 29.09733e9 # from microwave spectrum 29.09733(4)GHz (Hz)
        
        self.ThF_bool = True

class k_Hf:
    'Defining constants'
    def __init__(self):
        
        # Fundamental
        self.fund_c = 299792458 # speed of light [m/s]
        self.fund_kB = 1.3806488e-23 # Boltzmann constant [J/K]
        self.fund_epsilon0 = 8.845e-12 # permittivity [F/m]
        self.fund_mu0 = 4*pi*1e-7 # permeability [gauss m/A]
        self.fund_hbar = 1.05457173e-34 # hbar [J s]
        self.fund_e = 1.60217657e-19 # electron charge [C]
        self.fund_a0 = 5.29177210e-11 # Bohr radius [m]
        self.fund_uB = 9.274009994e-24 # Bohr magneton [J/T]
        self.fund_uN = 5.05050783888e-27 # nuclear magneton [J/T]
        self.fund_in = 0.0254 # inch in metres [m]
        
        # Conversion
        self.conv_Debye_to_SI = 3.33564e-30 # [C m / D]
        
        # HfF
        self.XF_m = 199 * 1.66053892e-27 # [kg]
        self.XF_charge = 1 # [of e]
        self.XF_A_par = -62.2 * 1e6 # hyperfine A_parallel [Hz]
        self.XF_omega_ef = 0.740 * 1e6 # Ω doubling [Hz]
        self.XF_d_mf = 3.567 # molecular frame dipole moment [D]; measured 23.8.2018
        self.XF_Lambda = 2 # 3D1
        self.XF_gN = 5.25774 # [unitless] that of F-19 (N. J. Stone, At. Data Nucl. Data Tables 90, 75 (2005))
        self.XF_gF = -0.003 # HfF+ [unitless]
        self.XF_Gpar = -3*self.XF_gF + self.XF_gN * self.fund_uN/self.fund_uB # [unitless] only holds for 3∆1
        self.XF_Joffset = (8.983*4)*1e9 # from Gen. I paper (Hz)
        
        self.ThF_bool = False

class p:
    'Defining parameters'
    def __init__(self,kk=None,boost=None,period_num=None,f_exp=None,t_step=None,loadTrap=None):
        
        # Boost factor to speed things up
        if boost==None:
            self.boost = 1
        else:
            self.boost = boost
        
        # Number of Ramsey periods to simulate
        if period_num==None:
            self.period_num = 1
        else:
            self.period_num = period_num
        
        # Boolean to check if Schrödinger's Equation has been solved
        self.ss_check = False
        
        # Switches
        self.BR_list = [ [B,R] for B in [1,-1] for R in [1,-1]] # vary LSB first, R is LSB, 1 then -1
        self.BDR_list = [ [B,D,R] for B in [1,-1] for D in [1,-1] for R in [1,-1]] # vary LSB first, R is LSB, 1 then -1
        
        # Modes for generating fields
        self.E_field_mode = 'ideal'
        self.B_field_mode = 'ideal'
        
        # Paul trap
        self.f_rf = 55e3 # (Hz)
        self.U0 = 2.3 # end cap voltage (V)
        self.V_rfpp = 12.2*2 # Vpp fin voltage (V)
        self.Vp = 0 # deconfining quadrupole fin voltage (V)
        
        # Erot
        if kk.ThF_bool==True:
            self.f_rot = 150e3 # (Hz)
        else:
            self.f_rot = 375e3 # (Hz)
        self.E_rot = 58e2 # [V/m] can take in array
        self.E_rot_phase = array([0,45,90,135,180,225,270,315])+202.5#+45 # extra 202.5º so that Erot points +x at t=0; extra 45 to symmetrize xpp and ypp
        self.E_rot_ramp = False
        self.E_rot_ramp_time = 10/self.f_rot # (s)
        
        # Time steps
        self.t_i = 0 # initial time of simulation (s)
        if f_exp==None:
            self.t_f = 1/15/self.boost*self.period_num # final time of simulation (s)
        else:
            self.t_f = 1/f_exp*self.period_num
        if t_step==None:
            self.t_step = 1/self.f_rot / 10 / ceil(1/self.f_rot/10 / 2.5e-7) # size of each time step (maximum size of 2.5e-7 s
        else:
            self.t_step = t_step
        self.motion_atol = 1e-4
        self.motion_rtol = 1e-4
        
        # Solver options
        self.num_chunk = 10
        self.num_chunk_debug = 1
        self.debug = False

        # Erot harmonics
        self.V_2harm_bool = False
        self.V_2harm_att = 0
        self.V_2harm_phase = 0
        self.V_2harm_a = cos(pi/2-pi/8)/cos(pi/8)
        self.E_displace_ramp = False
        
        # Edit
        self.V_dit_bool = False
        self.V_dit_phase = 0
        self.f_dit = 15e3 # [Hz]
        self.E_dit = 0.5e2 # [V/m]
        self.E_rot_a = 0.0268 # [1] for 50 Hz
        
        # Erot ellipticity
        self.E_rot_ellip_bool = False
        self.E_rot_ellip_e = 0 # (1+e) along x and (1-e) along y
        
        # Baxgrad (Gen. II)
        """
            For Erot=5700V/m, Baxgrad=23mG/cm, frot=375kHz, expect Brot=1.145e-7T, giving f0~14.4Hz

                Brot = Baxgrad * (e Erot) / (m omega_rot^2)
        """
        self.B_axial_grad_reversing = 2.3e-3*self.boost # [T/m] B = Baxgrad (x,y,-2z)
        self.B_axial_grad_nonreversing = 0 # [T/m] ambient field gradient of form (x,y,-2z)
        
        # Uniform B-fields
        self.B0_nr = array([0,0,0]) # [T] non-B-switching uniform fields, e.g. ambient fields
        self.B0 = array([0,0,0]) # [T] B-switching uniform fields, e.g. uniform fields induced by current-carrying coils
        
        # Other B-field gradients
        """
            Big Paper，eq 81, most general B field that satisfies div(B)=0, curl(B)=0:
            
                B = Baxgrad * (x, y, -z)
                    + Btrans * (x, -y, 0)
                    + B1 * (y, x, 0)
                    + B2 * (z, 0, x)
                    + B3 * (0, z, y)
        """
        self.B_trans = 0 # (x,-y,0) (T/m)
        self.B1 = 0 # (y,x,0) (T/m)
        self.B2 = 0 # (z,0,x) (T/m)
        self.B3 = 0 # (0,z,y) (T/m)
        
        # Imperfect anti-helmholtz coil [Gen. II, used with 'antihelm' mode]
        self.antihelm_I = 1*self.boost # current [A]
        self.antihelm_N = 86 # number of coils
        self.antihelm_R = 0.13 # radius (m)
        self.antihelm_d = 0.20 # coil separation (m)
        
        
        if loadTrap==None:
            # Spherical harmonics expansion for E-field [Gen. II, used with 'expansion' mode]
            self.lmax = 5
            if len(self.E_rot_phase) == 6:
                self.comsol_exp = loadmat('Ion Trap Data/multipole_data') # Import COMSOL numbers
            else:
                self.comsol_exp = loadmat('Ion Trap Data/multipole_data_gen2.mat')
            
            # Interpolation for E-field [Gen. II, used with 'interpolation' mode
            self.E_interp = load('Ion Trap Data/E_gen2_symZ.npy') # 10(fin)*3(dir)*61(x)*61(y)*101(z)
            self.E_interp_reorder = moveaxis(self.E_interp,[0,1],[-2,-1]) # 61(x)*61(y)*101(z)*10(fin)*3(dir)
        self.xs = arange(-0.03,0.031,0.001)
        self.ys = arange(-0.03,0.031,0.001)
        self.zs = arange(-0.05,0.051,0.001)
        
        # Trap parameters [Gen. II]
        """
            Huanqian's thesis eq 4.18 uses slightly different convention
            Conventions used are from "doi: 10.1063/1.367318"
            Also, kappa and R_eff are recalculated with COMSOL
        """
        self.trap_R_eff = 0.0493 # effective trap radius (m); from COMSOL
        self.trap_kappa = 0.21836 # end cap の geometric factor; from COMSOL
        self.trap_Z0 = 0.08 # half trap height (m); fixed
        self.trap_E_field_per_V_DC = array([17.5*0.787,17.5*0.787,1.5*0.977]) # V applied to E-field conversion ([(V/m trap) / (V DAC)]) 实验室里 LabVIEW 用的 conversion factor；并不一定是真的 conversion factor。真的由 COMSOL multipole expansion。
        self.trap_E_field_per_V_Erot = 17.5*0.9990404379310345 # checked numerically.
        
        # Kinematics (Ion Cloud)
        self.ion_cloud_T = 10 # [K]
        self.r0 = array([0,0,0])
        self.v0 = array([1,1,1]) * sqrt( kk.fund_kB * self.ion_cloud_T / ( 1/2 * kk.XF_m ) ) # kT = 1/2 mv^2; maximum position displacement agrees with kT = 1/2 m omega^2 r_max^2

    # Defining time steps used for calculation
    def gen_time_steps(self, t_i=None, t_f=None, t_step=None):
        """
            Use pre-set values if user does not give input
            Else, set pre-set values to user given input
        """
        if t_i == None:
            t_i = self.t_i
        else:
            self.t_i = t_i
        if t_f == None:
            t_f = self.t_f
        else:
            self.t_f = t_f
        if t_step == None:
            t_step = self.t_step
        else:
            self.t_step = t_step
            
        self.time_steps = arange(t_i,t_f,t_step)
        
        self.ts_check = True # Turned False by <Ramsey>.gen_motion; indicate to (re)generate motion

class q:
    'Defining the quantum system'
    def __init__(self,minJ=1,maxJ=1,Joffset=None,HrotSwitchBool=True,handle=None):
        # Importing newly defined k and p classes
        if handle==None:
            kk = k()
            pp = p()
        else:
            kk = handle.k
            pp = handle.p

        if maxJ==2 and Joffset==None:
            Joffset = kk.XF_Joffset
        else:
            Joffset = 0
        
        """
            Wigner-3j defined as (Brown & Carrington eqs 5.83/5.84)
                |J=0,M=0> = \sum_{m1,m2,m3} wigner_3j(j1,j2,j3,m1,m2,m3) |j1,m1;j2,m2;j3,m3>
              such that
                wigner_3j(j1,j2,j3,m1,m2,m3) = (-1)^(j3-m3) / sqrt(2*j3-1) <j1,m1;j2,m2|j3,-m3>
                                             = | j1 j2 j3 |
                                               | m1 m2 m3 |_3
                                               
              两个 j 加在一起给一个 j1+j2=j3 的意思。
            
            Wigner-6j defined as (Brown & Carrington eq 5.91)
                wigner_6j(j1,j2,j3,j4,j5,j6) = \sum_{a,b,c,d,e,f} (-1)^(j4+j5+j6+d+e+f)
                                                                  * wigner_3j(j1,j2,j3,a,b,e)
                                                                  * wigner_3j(j1,j5,j6,a,c,-f)
                                                                  * wigner_3j(j4,j2,j6,-d,b,f)
                                                                  * wigner_3j(j4,j5,j3,d,c,e)
                                             = | j1 j2 j3 |
                                               | j4 j5 j6 |_6
                      
              三个 j 加在一起给一个 j1+j2+j3，但是加的次序有讲究：(j1+j2)+j3、j1+(j2+j3)、(j1+j3)+j2。
              
            Useful properties of Wigner d-matrix:
                - (B&C 5.45) D^(j)_{m,m'}(a,b,c) = <j,m|exp(-i Jz a) exp(-i Jy b) exp(-i Jz c)|j,m'> = exp(-i(m a + m' c)) d^(j)_{m,m'}(b)
                - (B&C 5.99) \int dc sin(b) db da D^(j)_{m,k}(a,b,c) D^(j')_{m',k'}(a,b,c) = 8pi^2/(2j+1) delta(j,j') delta(m,m') delta(k,k')
                - (B&C 5.100) \int dc sin(b) db da D^(j1)_{m1,k1}(a,b,c) D^(j2)_{m2,k2}(a,b,c) D^(j3)_{m3,k3}(a,b,c) = 8pi^2 wigner_3j(j1,j2,j3,m1,m2,m3) wigner_3j(j1,j2,j3,k1,k2,k3)
                   Note: useful for proving Wigner-Eckart theorem
                   
            For finding matrix elements of tensor V of rank 1 (B&C 5.123/172)
                <j',m'|T^1_p(V)|j,m> = (-1)^(j'-m') * wigner_3j(j',k,j,-m',q,m) <j'||T^1_p(V)||j>
        """
        ### Columns of [J Omega F mF] for J={1,2}, Omega={-1,1} for 3D1, F={1/2,3/2}, mF={-F,...,F}
        self.q_nums = array([ 
            [J,Omega,F,mF] for J in range(minJ,maxJ+1) 
            for Omega in [-1,1] 
            for F in [J-1/2,J+1/2] 
            for mF in arange(-F,F+0.1,1)
        ])
        
        ### gF
        """
            Refer to Zeeman_elec below:
                g = -Gpar [F(F+1)+J(J+1) -3/4]/[2F(F+1)J(J+1)] + gN µN/µB [F(F+1)-J(J+1)+3/4]/[2F(F+1)]
        """
        self.gF = array([[(
            - kk.XF_Gpar * ( q[2]*(q[2]+1) + q[0]*(q[0]+1) - 3/4 ) / ( 2 * q[2]*(q[2]+1) * q[0]*(q[0]+1) )
            + kk.XF_gN * kk.fund_uN/kk.fund_uB * ( q[2]*(q[2]+1) - q[0]*(q[0]+1) + 3/4 ) / ( 2*q[2]*(q[2]+1) ))
            * delta(q[2],qp[2]) # no coupling across F
            for q in self.q_nums ] # unprimed across row
            for qp in self.q_nums ]) # primed down column
        
        ### Initial state
        self.psiUp_upper = Qobj([ 
            [((J==1) & (Omega==-1) & (F==3/2) & (mF==3/2))] for J in range(minJ,maxJ+1)
            for Omega in [-1,1] 
            for F in [J-1/2,J+1/2] 
            for mF in arange(-F,F+0.1,1)
        ])
        self.psiDown_upper = Qobj([ 
            [((J==1) & (Omega==1) & (F==3/2) & (mF==-3/2))] for J in range(minJ,maxJ+1)
            for Omega in [-1,1] 
            for F in [J-1/2,J+1/2] 
            for mF in arange(-F,F+0.1,1)
        ])
        self.psiUp_lower = Qobj([ 
            [((J==1) & (Omega==1) & (F==3/2) & (mF==3/2))] for J in range(minJ,maxJ+1)
            for Omega in [-1,1] 
            for F in [J-1/2,J+1/2] 
            for mF in arange(-F,F+0.1,1)
        ])
        self.psiDown_lower = Qobj([ 
            [((J==1) & (Omega==-1) & (F==3/2) & (mF==-3/2))] for J in range(minJ,maxJ+1) 
            for Omega in [-1,1] 
            for F in [J-1/2,J+1/2] 
            for mF in arange(-F,F+0.1,1)
        ])
        self.psi0_upper = (self.psiUp_upper + self.psiDown_upper) / sqrt(2)
        self.psi0_lower = (self.psiUp_lower + self.psiDown_lower) / sqrt(2)
        self.psi0 = [self.psi0_upper,self.psi0_lower]
        
        ### Sigma operators involving only the stretched states in the big Hilbert space
        self.Sx_upper = self.psiUp_upper * self.psiDown_upper.dag() + self.psiDown_upper * self.psiUp_upper.dag()
        self.Sy_upper = -1j * self.psiUp_upper * self.psiDown_upper.dag() + 1j * self.psiDown_upper * self.psiUp_upper.dag()
        self.Sz_upper = self.psiUp_upper * self.psiUp_upper.dag() - self.psiDown_upper * self.psiDown_upper.dag()
        self.Sx_lower = self.psiUp_lower * self.psiDown_lower.dag() + self.psiDown_lower * self.psiUp_lower.dag()
        self.Sy_lower = -1j * self.psiUp_lower * self.psiDown_lower.dag() + 1j * self.psiDown_lower * self.psiUp_lower.dag()
        self.Sz_lower = self.psiUp_lower * self.psiUp_lower.dag() - self.psiDown_lower * self.psiDown_lower.dag()
        self.Sx = [self.Sx_upper,self.Sx_lower]
        self.Sy = [self.Sy_upper,self.Sy_lower]
        self.Sz = [self.Sz_upper,self.Sz_lower]
        
        ### I = 1/2 for Fluorine-19
        self.I = 1/2
        
        ### Ω = J.n
        self.Omega = array([[ q[1] * delta(q[1],qp[1])
                             for q in self.q_nums ]
                            for qp in self.q_nums])
        """
            Omega_x is used for Ω-doubling part of Hamiltonian
            Ω-doubling only couples |Ω> and |-Ω> with the same {J,F,mF}
            Will's effective Hamiltonian reduces to ~sigma_x for J=1, with ~J^4 scaling, which is wrong!
                Will used [J(J+1)/2]^2, which is OK for J=1, but wrong for J=2
                Real scaling is ~J^2, so modified Hamiltonian to be J(J+1)
            Eq 1 in Dan's paper showed the true form of Ω-doubling.
                Since we define omega_ef as the splitting between the levels, and splitting according to Dan is k'J(J+1)+k"[J(J+1)]^2,
                then for k" too small and J=1, omega_ef=2k'.
            
            More details of Ω-doubling can be found in Amar Chandra Vutha's thesis Section A.4. Also see Dave DeMille's chapter 7.9 in "Atomic Physics" 2ed. In short:
                Hamilton of rotation of molecule is:
                    H_r = B N^2 , N is the rotation number operator of the molecule
                        = B (J - J_e)^2 , J,J_e are the total angular momentum of the system and electron, respectively.
                        = B J^2 + B J_e^2 - 2 B J.J_e,
                    first term gives the rotational lines
                    second term is a constant for a given electronic system
                    third term gives rise to Ω-doubling, also known as the Coriolis coupling term, H_Cor.
                    
                    H_Cor = -2B(Jx Jx_e + Jy Jy_e + Jz Jz_e) , defining z to be along n with N perp. to n, Jz = Jz_e = Ω
                          = -2B(- J^+ J^-_e - J^- J^+_e + Ω^2) , where J^± is spherical tensor notation: J^+ := -J_+/√2, J^- := J_-/√2, J_± := Jx ± i Jy.
                          = constant + 2B(J^+ J^-_e + J^- J^+_e)
                    
                    Without H_Cor, states of the same |Ω| are degenerate.
                    H_Cor couples states of neighbouring Ω, i.e. Ω to Ω±1. 
                    Expanding this out in perturbation theory, and noting the slight asymmetry in how the Ω couples, we see Ω-doubling.
                
            Extras on Lambda-douling (closely related, but not relevant here)
                For a Hund's case (a) state (H 3∆1 of ThO, X 3∆1 of ThF+), where spin-orbit coupling is not strong, Ω-doubling can be treated as Lambda-doubling.
                In the text, 
                    for a |Lambda|=1, the scaling with J goes as ~J^2 due to 2nd order perturbation theory coupling Lambda=1 to Lambda=-1, and only one J^± in each order.
                    for |Lambda|=2, would need 4th order perturbation theory.
                    
                However, in Ed Meyer's thesis, Lambda-doubling goes as (H_rot + H_SO)^(2 Lambda), where H_rot(N=J-s)
                So,
                    Scaling which couples Omega=±1 goes as J_+^2 S_+^2 + J_-^2 S_-^2 ~ J^2
                    
                B&C Section 7.4.5 says that Lambda-doubling comes from mixing with the Lambda=Sigma states, where each rotational level has non-degenerate parity eigenstates.
                
            Parity ordering:
                For 3∆1, J=1, the positive parity is energetically lower than negative parity.
                Since |Ω±> = (|+>±|->)/√2, the eigensolution of Ωx |Ω> will give a (±) eigenvalue for |Ω±>, i.e. positive parity more energetic than negative.
                To tackle this, it seems like using `-Ωx |Ω>` instead will do, but parity ordering switches with subsequent J's
                So the solution will be to use `(-1)^J Ωx |Ω>` instead.
        """
        self.Omega_x = array([[q[0]*(q[0]+1)/2 # J^2 scaling
                              * (-1)**q[0] # parity ordering (parity + sign has lower energy)
                              * not_zero(q[1]) # no Ω-doubling for Ω=0
                              * delta(q[1],-qp[1]) # coupling only |Ω> and |-Ω>
                              * delta(q[2],qp[2]) # F==F'
                              * delta(q[3],qp[3]) # mF==mF'
                              * delta(q[0],qp[0]) # J==J'
                              for q in self.q_nums ] # unprimed across row
                              for qp in self.q_nums ]) # primed down column
        
        ### F = I+J (vector sum)
        self.mF = diag(self.q_nums[:,3])
        """
            F_± only changes mF, not {F,J,Ω}
            F_±|F,mF> = √[(F-±mF)(F±mF+1)]|F,mF±1>, from usual ladder operator formalism
        """
        self.F_p = array([[sqrt((q[2]-q[3])*(q[2]+q[3]+1))
                          * delta(q[2],qp[2]) # F==F'
                          * delta(q[3]+1,qp[3]) # mF+1==mF'
                          * delta(q[0],qp[0]) # J==J' subspace
                          * delta(q[1],qp[1]) # Ω==Ω' subspace
                          for q in self.q_nums ] # unprimed across row
                          for qp in self.q_nums ]) # primed down column
        self.F_m = array([[sqrt((q[2]+q[3])*(q[2]-q[3]+1)) 
                          * delta(q[2],qp[2]) # F==F'
                          * delta(q[3]-1,qp[3]) # mF-1==mF'
                          * delta(q[0],qp[0]) # J==J' subspace
                          * delta(q[1],qp[1]) # Ω==Ω' subspace
                          for q in self.q_nums ] # unprimed across row
                          for qp in self.q_nums ]) # primed down column
        self.F_x = (self.F_p+self.F_m)/2
        self.F_y = (self.F_p-self.F_m)/2j
        
        ### S,∑,Lambda
        self.S = 1
        self.Sigma = -1
        self.Lambda = 2
        
        ### Columns of [F mF J I mJ mI Omega]
        blah = array([
            [F,mF,J,self.I,mJ,mI,Omega]
            for F in arange(1/2,(2*maxJ+1)/2+0.1,1)
            for mF in arange(-F,F+0.1,1)
            for J in range(1,maxJ+1)
            for mJ in arange(-J,J+0.1,1)
            for mI in arange(-self.I,self.I+0.1,1)
            for Omega in [-1,1] # for parity
        ])
        self.q_numsCG = blah[logical_and(
                                blah[:,1]==blah[:,4]+blah[:,5], # keep only mF = mJ+mI
                                logical_and(
                                    blah[:,0] <= blah[:,2]+blah[:,3],
                                    blah[:,0] >= abs(blah[:,2]-blah[:,3])
                                ) # keep only |J-I| <= F <= J+I
                            )
                        ]
        del blah
        
        ### E1 transition in between J's
        """
        E1 transition matrix across J's, not within J's
        
            1. Break |J Ω F mF> up into sum{ coeff |J I mJ mI Omega> } with self.F_to_J_I
            2. Evaluate <J'=2 I mJ' mI' Omega'| Sph_Ten(j=1 , p=±1,0) |J=1 I mI Omega> with self.E1_J_I
            3. Transition is given by
                <J' Ω' F' mF'| self.F_to_J_I.T * self.E1_J_I * self.F_to_J_I |J Ω F mF>
                = <J' Ω' F' mF'| self.E1 |J Ω F mF>
        """
        
        self.F_to_J_I = array([[clebsch_gordan(qp[2],qp[3],qp[0],qp[4],qp[5],qp[1])
                            * delta(q[0],qp[2]) # J's must agree
                            * delta(q[2],qp[0]) # F's must agree
                            * delta(q[3],qp[1]) # mF's must agree
                            * delta(q[1],qp[6]) # Ω's must agree (for parity)
                            for q in self.q_nums ] # unprimed across row
                            for qp in self.q_numsCG ]).astype(float) # primed down column
        
        """
            Want parity |±> to |-±> to be allowed but |±> to |±> disallowed.
            Such a selection rule matrix, would be a sigma_x in parity space.
            For |Ω> selection rules, note that
            
                |Ω±> = (|+> ± |->)/√2,
            
            so that |Ω±> = Hadamard |±>
            So selelection rules matrix in |Ω> space will be
            
                Hadamard sigma_x Hadamard = sigma_z.

            B&C eq 8.433 for Stark Effect, but E1 selection rules are the same
            <J' Ω M_J|-d.E|J Ω M_J> = -d E (-1)^(J'-M_J) wigner_3j(J',1,J,-M_J,0,M_J)
                                       * (-1)^(J'-Ω) * wigner_3j(J',1,J,-Ω,0,Ω)
                                       * √[(2J'+1)(2J+1)]
            
            Intensities still do not agree with experiment.
        """
        
        self.E1_J_I = array([[[(-1)**(qp[2]-qp[4]+qp[2]-qp[6]) * sqrt((2*qp[2]+1)*(2*q[2]+1)) 
                               * sp.N(wigner_3j(qp[2],1,q[2],-qp[4],p,q[4])) 
                               * sp.N(wigner_3j(qp[2],1,q[2],-qp[6],0,q[6]))
                               * delta(q[5],qp[5]) # mI's have to agree
                               * delta(q[6],qp[6]) * q[6] # Ω's have to agree (for parity), and is actually a sigma_z
                               * delta(q[2],1) * delta(qp[2],2) # for a J=1 to J=2 transition
                               for q in self.q_numsCG ] # unprimed across row
                              for qp in self.q_numsCG ] # primed down column
                             for p in [1,0,-1] ]) # for all polarizations    
    
        # For pedagogical purpose, Mickey Mouse version
        self.E1_F = array([ridZero(
                            [[clebsch_gordan(1,q[2],qp[2],p,q[3],qp[3])
                                 * delta(q[1],qp[1]) * q[1]  # Ω's have to agree (for parity), and is actually a sigma_z
                                 * delta(q[0],1) * delta(qp[0],2) # for a J=1 to J=2 transition
                                 for q in self.q_nums ] # unprimed across row
                                 for qp in self.q_nums ] # primed down column
                            ) 
                             for p in [1,0,-1] ]) # for all polarizations   
        
        """
            self.E1.6j and self.E1 assumes that the E1 dipole operature commutes with the nuclear spin,
            and acts on F through J only.
        """
        
        # Using Wigner 6-j symbol directly
        """
            From DeMille's book eq I.26,
            < J' I F' || T^k || J I F > = (-1)^(J'+I+F+k) * √[(2F+1)(2F'+1)] 
                                          * wigner_6j(J',F',I,F,J,k)
                                          * < J' || T^k || J >
            
            So just evaluate this like < J' I F' | T^k | J I F > acting on F directly, but include the relevant factors above.
        """
        self.E1_6j = array([ridZero(
                            [[clebsch_gordan(1,q[0],qp[0],p,q[1],qp[1])
                              * sqrt((2*q[2]+1)*(2*qp[2]+1))
                              * (-1)**(self.I+1+q[0]+qp[2])
                              * sp.N(wigner_6j(qp[0],qp[2],self.I,q[2],q[0],1))
                              * delta(q[1],qp[1]) * q[1]  # Ω's have to agree (for parity), and is actually a sigma_z
                              * delta(q[0],1) * delta(qp[0],2) # for a J=1 to J=2 transition
                              for q in self.q_nums ]
                              for qp in self.q_nums ]
                            )
                           for p in [1,0,-1] ]) # for all polarizations
        
        # Breaking up F to J,I and using Clebsch-Gordan
        self.E1 = array([ ridZero( self.F_to_J_I.T @ each.astype(float) @ self.F_to_J_I ) for each in self.E1_J_I ])

        ### J=1 and J=2 offset, in angular frequency (Hz)
        self.offset = array([[(q[0]-1)*(qp[0]-1)*Joffset # todo: change to B·J·(J+1)
                       * delta(q[1],qp[1]) * delta(q[2],qp[2]) * delta(q[3],qp[3])
                       for q in self.q_nums ] # unprimed across row
                       for qp in self.q_nums ] # primed down column
                       )
        
        ### Hyperfine (A_{parallel} J.I)
        """
            E_{HF} = A_{//} I.n J.n
                   = A_{//} I.n Ω
                   = A_{//} <J,Ω|I.n Ω|J,Ω>
                   = A_{//} Ω <J,Ω|I.J|J,Ω><J,Ω|J.n|J,Ω> / [J(J+1)] , projection theorem, Sakurai 3.10.40
                   = A_{//} Ω^2 <J,Ω|(F^2 - I^2 - J^2) / 2J^2|J,Ω>
                   = A_{//} (F^2 - I^2 - J^2) / 2J^2 , Ω=±1 ==> Ω^2 = 1
                   
            ∆E_{HF,J=1} / ∆E_{HF,J=2} = 9/5.
        """
        self.HF = diag((self.q_nums[:,2]*(self.q_nums[:,2]+1)
                        - self.q_nums[:,0]*(self.q_nums[:,0]+1)
                        - self.I*(self.I+1)) / (2*self.q_nums[:,0]*(self.q_nums[:,0]+1)))
        
        ### Stark (spherical tensor of rank 1)
        """
            Express E-field from lab frame in terms of molecule frame; E is spherial tensor of rank 1, i.e. k=1
            From B&C p.167, staring from 5.144
            
            p refers to frame of quantization axis, which defines the z-direction, i.e. mF, mJ. Erot is defined in p frame.
            q refers to frame of internuclear axis, which defines the molecular frame, i.e. Ω. d_mf is defined in q frame.

            By Wigner-Eckart theorem (p in frame of quantization axis, q in molecule frame, spherical tensor index)
                <F',mF'|D_{pq}^*(omega)|F,mF> = (-1)^(F'-mF') * wigner_3j(F',1,F,-mF',p,mF) * <F'||D_{pq}^*(omega)||F> , (B&C 5.123/172)
                  = (-1)^(F'-mF') * wigner_3j(F',1,F,-mF',p,mF) * <J',I',J'+I'=F'||D_{.q}^*(omega)||J',I,J+I=F>

                    Split composite system
                    <J',I',J'+I'=F'||D_{.q}^*(omega)||J',I,J+I=F> = delta(I,I') * (-1)^(F+J'+1+I') * √[(2F'+1)(2F+1)] * wigner_6j(J,F,I',F',J',1) * <J'||D_{.q}(omega)||J> , (B&C 5.174)

                    Double-bar matrix element
                    <J'||D_{.q}^*(omega)||J> = (-1)^(J'-Ω') √((2J'+1)(2J+1)) wigner_3j(J' 1 J; -Ω' q Ω) (B&C 5.148)

                  = (-1)^(F'-mF') * wigner_3j(F',1,F,-mF',p,mF)
                    * delta(I,I') * (-1)^(F+J'+1+I) √((2F+1)(2F'+1)) wigner_6j(J F I; F' J' 1)
                    * (-1)^(J'-Ω') √((2J'+1)(2J+1)) wigner_3j(J' 1 J; -Ω' 0 Ω), q=0 because dipole is aligned with internuclear axis.
                    
            Note from above, states of same Ω,mF across different F's and J's can couple, giving a difference in g_F for the upper and lower doublets.
        """
        self.Stark_sph_ten = array([[[(-1)**(qp[2]-qp[3]) * sp.N(wigner_3j(qp[2],1,q[2],-qp[3],p,q[3]))
                                      * (-1)**(q[2]+qp[0]+1+self.I) * sqrt((2*q[2]+1)*(2*qp[2]+1)) * sp.N(wigner_6j(q[0],q[2],self.I,qp[2],qp[0],1))
                                      * (-1)**(qp[0]-qp[1]) * sqrt((2*qp[0]+1)*(2*q[0]+1)) * sp.N(wigner_3j(qp[0],1,q[0],-qp[1],0,q[1]))
                                      for q in self.q_nums ] # unprimed across row
                                     for qp in self.q_nums ] # primed down column
                                    for p in [1,0,-1] ]).astype(float64)
        
        ### Zeeman term for all electronic contributions (orbital and spin)
        """
            From Petrov's arXiv:1704.06631, we have
            
                Eq3: g = -Gpar [F(F+1)+J(J+1) -3/4]/[2F(F+1)J(J+1)] + gN µN/µB [F(F+1)-J(J+1)+3/4]/[2F(F+1)] = (-Gpar + gN µN/µB)/3 for F=3/2,J=1
                Eq8: H = µB (L - gS S).B - gN µN I.B
                Eq21:Gpar = <L - gS S>/Ω, expectation value taken at 3∆1
                
                Results: Gpar = 0.012043 (applicable to all states in 3∆1)
            
            For gF = -0.003 (HfF+) for F=3/2, J=1, gN = 5.25774 (F), we have Gpar = 0.011863.
            
            H = -µ.B = - ( -gL µB L + -gs µB S).B = µB ( gL L + gS S ).B
                In Petrov, gS is actually g_e, which is |gS(<0)|.
                Also, gL=1 for an infinite-mass nucleus.
            
            Using B&C p.606 9.57; neglecting q=±1 because these terms mix vibronic states; book only has the q=0 component, good. Book neglects p=±1 components though, as it assumes that B is pointing in the z-direction. Reintroducing all p components.
            
            In 9.57, identify Lambda (gL+gr) as Gpar.
            
            Although 9.57 is for orbital component, the equation for spin (9.58) is just -1 times 9.57. Since we are combining both L and S into L + gS S, we can just use 9.57 for the total electronic component.
            
            The same argument does not hold for the rotation component since we are only chucking L + gs S into Gpar and not the rotation part. But gr should be much smaller than everything else, so we can ignore it (for now).
            
            Expect answer to be the same as self.Stark_sph_ten since we are just switching out {d_mf , B} for {g µB , E}.
                
            self.Zeeman_elec = (-1)^(F'-mF'+F+J'+I+1+J'-Ω') wigner_6j(J' F' I; F J 1)
                                * wigner_3j(F' 1 F; -mF' p mF) * wigner_3j(J' 1 J; -Ω' 0 Ω)
                                * √((2F'+1)(2F+1)(2J'+1)(2J+1))
        """
        self.Zeeman_elec = array([[[(-1)**(qp[2]-qp[3]+q[2]+self.I+1-qp[1]) * sp.N(wigner_6j(qp[0],qp[2],self.I,q[2],q[0],1))
                                    * sp.N(wigner_3j(qp[2],1,q[2],-qp[3],p,q[3]))
                                    * sp.N(wigner_3j(qp[0],1,q[0],-qp[1],0,q[1]))
                                    * sqrt((2*qp[2]+1)*(2*q[2]+1)*(2*qp[0]+1)*(2*q[0]+1))
                                    for q in self.q_nums ] # unprimed across row
                                   for qp in self.q_nums ] # primed down column
                                  for p in [1,0,-1] ]).astype(float64)
        
        ### Zeeman nuclear (spherical tensor of rank 1, but only using the q=0 component)
        """
        Using B&C p.606 9.59; book only has the q=0 component, generalized to include all three components
            
            -gN µN Bz self.Zeeman_nucl
            
            self.Zeeman_nucl = (-1)^(F'-mF'+J'+I+F'+1) √(I(I+1)(2I+1)(2F'+1)(2F+1))
                                        * wigner_6j(I F' J'; F I 1) * wigner_3j(F' 1 F; -mF' p mF)
        """
        self.Zeeman_nucl = array([[[(-1)**(qp[2]-qp[3]+qp[0]+self.I+qp[2]+1)
                                    * sqrt(self.I*(self.I+1)*(2*self.I+1)*(2*qp[2]+1)*(2*q[2]+1))
                                    * sp.N(wigner_6j(self.I,qp[2],qp[0],q[2],self.I,1))
                                    * sp.N(wigner_3j(qp[2],1,q[2],-qp[3],p,q[3]))
                                    * delta(q[1],qp[1])
                                    for q in self.q_nums ] # unprimed across row
                                   for qp in self.q_nums ] # primed down column
                                  for p in [1,0,-1] ]).astype(float64)
        
        """
            arXiv:1704.07928v1 の eq (S7): (c.f. Big paper Eq10)

                H = H_tum + H_hf + H_Stark + H_Omega_doubling + H_rot + H_Zeeman_e + H_Zeeman_nucleus + H_edm

                H_tum = B_e J^2
                    B_e: rotational constant
                    J: L+S+R
                H_hf = A_parallel (I.n)(J.n)
                    A_parallel: hyperfine constant [-20.3MHz]
                    I: Fluorine nuclear spin (I_Th = 0)
                    n: F->Th internuclear axis
                H_Stark = -d_mf Omega gamma_F E [Big paper Eq20]
                    d_mf: molecular frame molecule dipole moment [2.74D PRA 91, 042504 (2015)]
                    Omega: ±1 for 3∆1
                    gamma_F: (J^2 + F^2 - I^2) / (2 F^2 J^2) [Big paper Eq21]
                    E: E_rot
                H_Omega_doubling = hbar omega_ef Omega_x/2
                    omega_ef: Omega doublet splitting [5.3MHz]
                    Omega_x: effective coupling operator that couples |Ω> to |-Ω> with the same {J,F,mF}
                H_rot = - hbar omega_rot.F
                    omega_rot: E_rot rotational (2π) frequency
                    F: I+J
                H_Zeeman_e = G_parallel (J.n)(B.n)
                    G_parallel: 3 gF - gN µN/µB
                    g_F: F=3/2,J=1 g-factor
                    B: B_rot
                H_Zeeman_nucleus = -gN µN I.B
                    g_N: g factor for fluorine nucleus [5.25774(2)]
                    mu_N: nuclear magneton
                H_edm = -d_e E_eff Omega
                    d_e: EDM
                    E_eff: calculated effective E-field
                    Omega: J.n
        """

        # Hyperfine [splitting], in angular frequency [Hz]
        self.H_hf = kk.XF_A_par * self.HF
        
        # Ω doubling [splitting], in angular frequency [Hz]
        self.H_Omega_doubling = kk.XF_omega_ef * self.Omega_x / 2
        
        # Rotation (Coriolis) [mF mixing], in angular frequency [Hz]
        """
            From arXiv:0909.2061v1, they investigate the Hamiltonian of a two-level system with basis in the lab frame {X,Y,Z}.
                - System has dipole moment pointing along a direction n, with internal axes {x,y,z} rotated by theta down from Z towards X.
                - n precesses around Z at angular frequency omega_rot
                - Moving into a rotating frame such that the Hamiltonian is time-independent gives the dressed Hamiltonian.
                - Turns out the dressed Hamiltonian has the form:
                    - H_dressed = H_non-rotating + H_rotating
                    - H_non-rotating has no information on omega_rot, and is the Hamiltonian of the system as though there is no rotation, just static, i.e., in the rotating frame.
                    - All information of omega_rot is contained in H_rotating
                - H_rotating has the form: - omega_rot sigma_Z
                - When using the {x,y,z} basis, it becomes: - omega_rot [ cos theta sigma_z - sin theta sigma_x ].
                - Since we are defining our quantization axis to be along the X-Y plane, theta=90deg.
        """
        self.H_rot = pp.f_rot * self.F_x # -omega_rot ( cos(theta) Fz - sin(theta) Fx )
        self.H_rot_switch = HrotSwitchBool

    # Rotation [mF-mixing] [Hz]
    def H_Rot(self,Rswitch,kk=None,pp=None):
        """
            F here is an operator defined w.r.t. the quantization axis, not the molecular axis, not the lab frame.
            - For Schrödinger integration, there is no alpha involved.
            - For quick diagonalization, however, we may need to introduce an alpha to mimic the geometric phase.
        """
        if kk==None:
            kk = k_Hf()
        if pp==None:
            pp = p(kk=kk)
        
        return pp.f_rot * self.F_x * Rswitch
    
    # Stark [parity mixing, mixes J and F of the same Ω,mF] [Hz]
    def H_Stark(self,E=None,kk=None,pp=None):
        """
            Allows for a switch between Erot and real E-field.
            See `self.Stark_sph_ten`.
        """
        
        if kk==None:
            kk = k_Hf()
        if pp==None:
            pp = p(kk=kk)
        
        if all(E)==None:
            # Use Erot, in angular frequency [Hz]
            return - kk.XF_d_mf*kk.conv_Debye_to_SI/kk.fund_hbar * self.Stark_sph_ten[1] * pp.E_rot /2/pi
        else:
            # Use real field, in angular frequency [Hz]
            return - kk.XF_d_mf*kk.conv_Debye_to_SI/kk.fund_hbar * inner_sph_cart(self.Stark_sph_ten,E) /2/pi

    # Zeeman [F-mixing] [Hz]
    def H_Zeeman(self,B,kk=None,pp=None):
        """
            Ideally we want H = - gF µB F.B, but we only measured gF(F=3/2) and we know nothing about gF(1/2). Fortunately we can relate gF(3/2) to gF(1/2).
            
            From Petrov's arXiv:1704.06631, we have
            
            Eq3: g = -Gpar [F(F+1)+J(J+1) -3/4]/[2F(F+1)J(J+1)] + gN µN/µB [F(F+1)-J(J+1)+3/4]/[2F(F+1)] = (-Gpar + gN µN/µB)/3 for F=3/2,J=1
            Eq8: H = µB (L - gS S).B - gN µN I.B
            Eq21:Gpar = <L - gS S>/Ω, expectation value taken at 3∆1
            
            H_Zeeman = Gpar µB J.n n.B - gN µN I.B, J.n=Ω
                There should also be a `Gperp µB J.n_perp n_perp.B` term, but this averages to zero given that Ω is a good quantum number, i.e., J precesses around n.
            
            Note: Following Gpar convention in Petrov, not Cairncross.
        """
        if kk==None:
            kk = k_Hf()
        if pp==None:
            pp = p(kk=kk)
        
        """
            Using only Bz because the ±1 components of the spherical tensors for Zeeman_elec and Zeeman_nucl are zero.
        """
        return inner_sph_cart( kk.XF_Gpar * kk.fund_uB * self.Omega * self.Zeeman_elec - kk.XF_gN * kk.fund_uN * self.Zeeman_nucl , B ) /kk.fund_hbar/2/pi
       

    # Altogether, [2π Hz]
    def big_H(self,r,t,kk,pp,BR=None):
        if BR==None:
            Bswitch,Rswitch = 1,1
        else:
            [Bswitch,Rswitch] = BR
        if len(pp.E_rot_phase)==6:
            E_mol = to_rot(E_field(self,r,t,kk,pp,Rswitch),pi/2,2*pi*pp.f_rot*t*Rswitch-pi/2)
            B_mol = to_rot(B_field(self,r,t,kk,pp,Bswitch),pi/2,2*pi*pp.f_rot*t*Rswitch-pi/2)
        else:
            E_mol = to_rot(E_field(self,r,t,kk,pp,Rswitch),pi/2,2*pi*pp.f_rot*t*Rswitch)
            B_mol = to_rot(B_field(self,r,t,kk,pp,Bswitch),pi/2,2*pi*pp.f_rot*t*Rswitch)
        return 2*pi* (self.H_hf
                      + self.H_Omega_doubling
                      + self.H_Rot(Rswitch,kk=kk,pp=pp)
                      + self.H_Stark(E_mol,kk=kk,pp=pp)
                      + self.H_Zeeman(B_mol,kk=kk,pp=pp)
                      + self.offset)
    
    # Hrot switch
    def HrotSwitch(self,E_rot,kk=None):
        """
            True: use comparator to turn on Hrot gradually w.r.t. omega_ef
            False: always turn on Hrot
        """
        if kk==None:
            kk = k_Hf()
        if self.H_rot_switch:
            return comparator(kk.XF_d_mf*kk.conv_Debye_to_SI/kk.fund_hbar/(2*pi)*E_rot , kk.XF_omega_ef)
        else:
            return 1


####------------ Helper functions ------------####


##---- Solvers for parfor ----##

# Solve kinematics, written for parfor
def kineSolve(Rswitch,kwarg={}):
    # For odeint, dy/dt given y and t
    def kine(t,y):
        """
            y is a vector of [r' r], with r = [x y z]
            scipy.integtrate.odeint solves a first order DE y'(t) = f(y,t)
            だから、need to transform the kinematics equation into a first order equation:

                r''(t) = eE(t)/m.
            
            Add a damping coefficient for the pre-motion to get rid of slosh:
            
                r''(t) = eE(t)/m - br'(t).

            Returns dy/dt = [ax ay az vx vy vz]
        """
        rd,r = y[0:3],y[3:6]

        if kwarg['self'].p.E_field_mode[:8]=='harmonic':
            yd = array([ - (2*pi*1.3e3)**2 * (r - kwarg['self'].p.r0*turn_on_ramp(t,kwarg['self'].p.E_rot_ramp,kwarg['self'].p.E_rot_ramp_time)) + kwarg['self'].k.XF_charge * kwarg['self'].k.fund_e * ( E_field(kwarg['self'],r,t,Rswitch=Rswitch) ) / kwarg['self'].k.XF_m - kwarg['self'].damp_b * rd, rd ])
        else:
            yd = array([kwarg['self'].k.XF_charge * kwarg['self'].k.fund_e * ( E_field(kwarg['self'],r,t,Rswitch=Rswitch) ) / kwarg['self'].k.XF_m - kwarg['self'].damp_b * rd, rd])

        return yd.flatten()
    
    """
        Previous version:
            - odeint
            - atol=rtol=1e-8
            - Adam's method
        This version:
            - solve_ivp
            - default settings at rtol=1e-3, atol=1e-6
            - RK5
    """
    sol = solve_ivp(fun=kine, t_span=[0,kwarg['kine_time_steps'][-1]], y0=kwarg['y0'][int((Rswitch-1)/-2)], t_eval=kwarg['kine_time_steps'], first_step=kwarg['self'].p.t_step,atol=kwarg['self'].p.motion_atol,rtol=kwarg['self'].p.motion_rtol)
    return sol.y

# Solve Schrödinger's equation by propagation, written for parfor
def propSolve(n_BR,kwarg={}):
    
    offset,num_chunk,total_step_per_chunk,self = kwarg['offset'],kwarg['num_chunk'],kwarg['total_step_per_chunk'],kwarg['self']
    
    # EDM Hamiltonian
    def H(t,args={'BR':[1,1]}):
        """
            `mesolve` takes in a H assuming that hbar is set to 1, so frequencies have to be expressed as angular frequencies.
        """
        return Qobj(self.q.big_H(self.simul_r[int((args['BR'][1]-1)/2)].T[int(round(t/self.p.t_step))],t,self.k,self.p,args['BR']))
    
    def asym(state,Dswitch_index):
        """
            ( |a|^2 - |b|^2 ) / ( |a|^2 + |b|^2 )
        """
        if Dswitch_index==0:
            psiUp,psiDown = kwarg['self'].q.psiUp_upper,kwarg['self'].q.psiDown_upper
        else:
            psiUp,psiDown = kwarg['self'].q.psiUp_lower,kwarg['self'].q.psiDown_lower
        a2 = abs(state.overlap((psiUp+psiDown)/sqrt(2)))**2
        b2 = abs(state.overlap((psiUp-psiDown)/sqrt(2)))**2
        return (a2-b2)/(a2+b2)
        
    def expects(state,Dswitch_index):
        """
            Calculates the expectation values of the Pauli matrices without normalization to the doublet subspace.
            Any Bloch vector with length less than 1 implies leakage of probability into non-doublet states.
        """
        return array([expect(kwarg['self'].q.Sx[Dswitch_index],state),
                      expect(kwarg['self'].q.Sy[Dswitch_index],state),
                      expect(kwarg['self'].q.Sz[Dswitch_index],state)])
    
    # Solve in chunks
    for chunk in range(offset,num_chunk+offset):
        
        # Get chunk of time steps
        start_index , end_index = total_step_per_chunk*(chunk) , total_step_per_chunk*(chunk+1)
        ts = self.p.time_steps[start_index:end_index]
        
        # Determine initial state
        if chunk==0:
            psi0 = self.q.psi0
        else:
            psi0 = self.simul_result_chunks_last_psi0[n_BR][chunk-1]
        print(str(self.p.BR_list[n_BR])+' @ '+'Chunk %d/%d:'%(chunk+1,num_chunk+offset))

        # Initialize array
        asyms = [ [] for each in ts ]
        exps = [ [] for each in ts ]
        asyms[0] = [ asym(psi0[Dswitch_index],Dswitch_index) for Dswitch_index in range(2) ]
        exps[0] = [ expects(psi0[Dswitch_index],Dswitch_index) for Dswitch_index in range(2) ]
        
        # Time step
        dt = ts[1] - ts[0]
        
        # Propogate step by step
        rep = False
        for n in range(len(ts)-1):
            
            # Print progress
            prog = floor((n+1)/(len(ts)-1)*100)
            if mod(prog,10)==0:
                if not rep:
                    datestring = datetime.datetime.now()
                    ds = datestring.strftime("%H:%M:%S")
                    print('('+ds+'): '+str(self.p.BR_list[n_BR])+' @ '+str(int(prog))+"%")
                    rep = True
            else:
                rep = False

            # Propagate state
            psi0 = [ (H(ts[n],args={'BR':self.p.BR_list[n_BR]})/1j*dt).expm() * state.unit() for state in psi0 ]
            
            # Calculate asymmetry and Pauli expectation values
            asyms[n+1] = [ asym(psi0[Dswitch_index],Dswitch_index) for Dswitch_index in range(2) ]
            exps[n+1] = [ expects(psi0[Dswitch_index],Dswitch_index) for Dswitch_index in range(2) ]
         
        datestring = datetime.datetime.now()
        ds = datestring.strftime("%H:%M:%S")
        print('('+ds+'): '+str(self.p.BR_list[n_BR])+' @ Done!')
        self.simul_result_chunks_asyms[n_BR][chunk] = asyms
        self.simul_result_chunks_exps[n_BR][chunk] = exps
        self.simul_result_chunks_last_psi0[n_BR][chunk] = [ (H(ts[-1],args={'BR':self.p.BR_list[n_BR]})/1j*dt).expm() * state.unit() for state in psi0 ] # propagate once more for the starting state for the next chunk
    
    return self.simul_result_chunks_asyms[n_BR],self.simul_result_chunks_exps[n_BR],self.simul_result_chunks_last_psi0[n_BR]


##---- Electrodynamics ----##

# Fin voltage from Paul trap E-field
def V_paul(t,kk,pp):
    """
        [1 2 3 4 5 6 7 8 T B with 1-8 in +x direction]
        
        With my conventions,
            V_rf = V_rfpp/2 * (x^2-y^2)/R_eff^2 * cos(2 pi f_rf t)
    """
    # 1 2 3 4 5 6 7 8 T B
    return append(pp.V_rfpp/2 * cos(2*pi*pp.f_rf*t) * array([1,-1,-1,1,1,-1,-1,1]) , pp.U0*ones(2))

# Fin voltage from rotating E-field
def V_rot(t,kk,pp,Rswitch):
    """
        [1 2 3 4 5 6 7 8 T B with 1-8 in +x direction]
    """
    # 1 2 3 4 5 6 7 8 T B, Erot sweeps CCW as seen from top with Rswitch=+1
    return append(pp.E_rot / pp.trap_E_field_per_V_Erot * cos(2*pi*pp.f_rot*-Rswitch*t + pp.E_rot_phase/180*pi) , zeros(2))

# Uniform field at second harmonic
def V_2harm(t,kk,pp,Rswitch):
    # 1 2 3 4 5 6 7 8 T B, Erot sweeps CCW as seen from top with Rswitch=+1
    amps = array([-1,-pp.V_2harm_a,pp.V_2harm_a,1,1,pp.V_2harm_a,-pp.V_2harm_a,-1])
    return append(pp.V_2harm_att * pp.E_rot / pp.trap_E_field_per_V_Erot * cos(2*pi*pp.f_rot*-Rswitch*t * 2 + pp.V_2harm_phase/180*pi) * amps , zeros(2))

# Mickey Mouse ramp
def turn_on_ramp(t,rampBool,ramp_time):
    if rampBool:
        return min(t,ramp_time)/ramp_time
    else:
        return 1

# Altogether
def V_tot(t,kk,pp,Rswitch):
    
    # Fin voltage from displacing E-field
    """
        Want to displace ions from origin in harmonic pseudopotential trap to arbitrary starting r0 with DC E-field
        This E-field (E_field_DC) also affects ion motion in trap から、so include in calculation
    """
    E_field_displace_DC = kk.XF_m * pp.trap_omega**2 / (kk.fund_e * kk.XF_charge) * pp.r0
    V_displace_DC_xyz = E_field_displace_DC / pp.trap_E_field_per_V_DC # convert E-field to V of electrodes
    """
        Gen. II: [fins 1 2 3 4 5 6 7 8 CCW with 1-8 in +x direction, top end cap, bottom end cap]
        Negative voltage to attract positive ions in positive direction
    """
    # 1 2 3 4 5 6 7 8 T B
    V_displace_DC = array([-V_displace_DC_xyz[0],-V_displace_DC_xyz[1],
                           -V_displace_DC_xyz[1],V_displace_DC_xyz[0],
                           V_displace_DC_xyz[0],V_displace_DC_xyz[1],
                           V_displace_DC_xyz[1],-V_displace_DC_xyz[0],
                          -V_displace_DC_xyz[2], V_displace_DC_xyz[2],])
    
    # return [1,0,0,0,0,0,0,0,0,0] # to debug which entry corresponds to which fin/EC
    return  ( V_paul(t,kk,pp) 
              + ( V_rot(t,kk,pp,Rswitch) + pp.V_2harm_bool * V_2harm(t,kk,pp,Rswitch) ) * turn_on_ramp(t,pp.E_rot_ramp,pp.E_rot_ramp_time) 
              + V_displace_DC * turn_on_ramp(t,pp.E_displace_ramp,pp.E_displace_ramp_time) )

# Generate E-field with V_tot (expansion)
def gen_E(r,V_ec_fins,kk,pp,lmax=None):
    """
        Will's method:
        1. Expand E-field generated by each fin given some potentials into spherical harmonics (COMSOL)
            (powers) 1,3,1,165 for column,xyz,nothing,a+b+c=l to l=8
            (coeffs) 1,3,100,165 for column,xyz,{l,m} to l=10,a+b+c to l=8
            (amps) 1,8,100 for column,6 fins and 2 endcaps,{l,m} to l=10,a+b+c to l=8
        2. Calculate time dependence of potential of each fin through:
            a) displacing E-field
            b) Paul trap E-field
            c) rotating E-field
        3. Sum all potentials together and calculate E-field spherical harmonics

        讨论：
        - 如果用的是 lattice E-field，那 trajectory 中所看到的 E-field 也被 lattice discretize，不一定最准
        - 如果用的是 spherical harmonic，那 trajectory 中看到的 E-field 都可以推出来，只差在 higher order。
    """
    if lmax == None:
        lmax = 3

    lmax = min(lmax,8) # Will's COMSOL expansion only accounts up to lmax=8

    """
        Number of a+b+c=l up to lmax:
            sum([(n+1)*(n+2)/2 for n in range(lmax+1)])
        The above from (n+1)*n/2 + (n+1):
            - Count with n balls and 2 partitions.
            - First term from number of distinct combinations of partitions in different spots (nC2)
            - Second term from number of distinct combinations of partitions in same spot (nC1)
    """
    num_xyz_abc_l = int(lmax**3/6 + lmax**2 + 11*lmax/6 + 1)

    """
        Number of {l,m} up to lmax
        sum([2*n+1 for n in range(lmax+1)])
    """
    num_lm = lmax**2 + 2*lmax + 1

    # Filter out required components
    powers = pp.comsol_exp['powers'][0,0:,:,:num_xyz_abc_l]
    coeffs = pp.comsol_exp['coeffs'][0,:,:num_lm,:num_xyz_abc_l]
    amps = pp.comsol_exp['amps'][0,:,:num_lm]

    # All the power combinations of x^a y^b z^c in Cartesian Spherical Harmonics
    before_coeffs = prod(array([ r[i] ** powers[i].flatten() for i in range(3)]),0) # shape = (num_xyz_abc_l,)

    # All the Cartesian Spherical Harmonics
    after_coeffs = array([[sum(before_coeffs * each) for each in eacheach] for eacheach in coeffs]) # shape = (3, num_lm)

    # All the amplitudes of the Spherical Harmonics of all end caps and 6 fins [t,b,1,2,3,4,5,6[,7,8]]
    after_amps = array([sum([each * V_ec_fins]) for each in amps.T]) # shape = (num_lm,)

    return array([sum(each * after_amps) for each in after_coeffs]) # shape = (3,)

# Generate E-field with V_tot (interpolation)
def gen_E_interp(r,V_ec_fins,kk,pp):
    """
        Using COMSOL to simulate the potential/fields due to each fin/end cap kept at 1V (with the rest kept at GND), the data is saved by MATLAB. Region evaluated is nominally $x,y\in[-3,3]$cm and $z\in[-5,5]$cm with step sizes of 1mm.

        MATLAB uses `ndgrid(x,y,z)` which iterates through `x` first and `z` last. All the data saved by MATLAB is in column matrix style, with each entry corresponding to an `ndgrid(x,y,z)`.

        Rearranging and exporting the data in a numpy friendly style, `E_gen2.py` is in:

            10(fins) of 3(axes) of 61(x)*61(y)*101(z)
            
        The above order is further rearranged to
        
            61(x)*61(y)*101(z) of 10(fins) of 3(axes)
            
        for RegularGridInterpolator
    """
    interp_fin_ec = RegularGridInterpolator((pp.xs,pp.ys,pp.zs),pp.E_interp_reorder)
    return sum(interp_fin_ec(r).reshape(10,3) * V_ec_fins.reshape(10,1) , axis=0)
    
# E-field
def E_field(self,r,t,kk=None,pp=None,Rswitch=None):
    
    if pp==None:
        pp = self.p
    if kk==None:
        kk = self.k
    if Rswitch==None:
        Rswitch=1

    if pp.E_field_mode == 'expansion':
        E_t = gen_E(r,V_tot(t,kk,pp,Rswitch),kk,pp,pp.lmax)

    elif pp.E_field_mode == 'interpolation':
        E_t = gen_E_interp(r,V_tot(t,kk,pp,Rswitch),kk,pp)

    elif pp.E_field_mode == 'harmonic':
        E_t = gen_E_interp(r, ( V_rot(t,kk,pp,Rswitch) + pp.V_others_bool * V_others(t,kk,pp,Rswitch) ) * turn_on_ramp(t,pp.E_rot_ramp,pp.E_rot_ramp_time),kk,pp) - (2*pi*1.3e3)**2 * (r - pp.r0*turn_on_ramp(t,pp.E_rot_ramp,pp.E_rot_ramp_time)) * kk.XF_m / kk.fund_e # include effects of the harmonic springs as E-fields

    elif pp.E_field_mode == 'harmonic_Ez':
        E_t = array([0,0,pp.Ez_net]) + gen_E_interp(r, ( V_rot(t,kk,pp,Rswitch) + pp.V_others_bool * V_others(t,kk,pp,Rswitch) ) * turn_on_ramp(t,pp.E_rot_ramp,pp.E_rot_ramp_time),kk,pp)
        
    elif pp.E_field_mode == 'rotating':
        E_t = pp.E_rot * turn_on_ramp(t,kk,pp) * ( 
            array(list(map(lambda x: x(2*pi*pp.f_rot*Rswitch*t),[cos,sin,ling]))) # Fundamental
        )

    elif pp.E_field_mode == 'cycloid':
        E_t = array([1,0,0]) / kk.fund_e * kk.XF_m # cycloid debug
        
    elif pp.E_field_mode == 'x':
        E_t = array([1,0,0]) / kk.fund_e * kk.XF_m # cycloid debug
    
    elif pp.E_field_mode == 'y':
        E_t = array([0,1,0]) / kk.fund_e * kk.XF_m # cycloid debug
    
    elif pp.E_field_mode == 'z':
        E_t = array([0,0,1]) / kk.fund_e * kk.XF_m # cycloid debug

    elif pp.E_field_mode == 'ideal':
        # For RF confinement
        E_paul_RF = - pp.V_rfpp/2 * cos(2*pi * pp.f_rf * t) * 2 * array([1,-1,0]) * r / pp.trap_R_eff**2
        E_paul_DC = - pp.U0 * pp.trap_kappa / pp.trap_Z0**2 * array([-1,-1,2]) * r
        
        # Erot and harmonics
        E_rot = pp.E_rot * array([ (1 + pp.E_rot_ellip_bool*pp.E_rot_ellip_e) * cos(2*pi * pp.f_rot * Rswitch * t) ,
                                   (1 - pp.E_rot_ellip_bool*pp.E_rot_ellip_e) * sin(2*pi * pp.f_rot * Rswitch * t) ,
                                   0 ]) * ( 1 - pp.E_rot_a * cos(2*pi * pp.f_dit * t) )
        E_2harm = pp.V_2harm_att * pp.E_rot * array([1,0,0]) * cos(4*pi * pp.f_rot * t + pp.V_2harm_phase/180*pi)
        
        # E_dit
        E_dit = pp.E_dit * array([ 0, 0, 1 ]) * cos(2*pi * pp.f_dit * t + pp.V_dit_phase/180*pi)
        
        # Total
       
        E_t = ( E_paul_RF 
                + E_paul_DC
                + E_rot * turn_on_ramp(t,pp.E_rot_ramp,pp.E_rot_ramp_time)
                + pp.V_2harm_bool * E_2harm * turn_on_ramp(t,pp.E_rot_ramp,pp.E_rot_ramp_time)
                + pp.V_dit_bool * E_dit * turn_on_ramp(t,pp.E_rot_ramp,pp.E_rot_ramp_time)
               )
        
    return E_t

# For antihelm B-field
def cosInt(A,B):
    if B==0:
        return 0
    else:
        return -4 * ( A*ellipe(2*B/(A+B)) + (-A+B)*ellipk(2*B/(A+B)) ) / ( (A-B)*B*sqrt(A+B) )

# For antihelm B-field
def sinInt(A,B):
    return 0;

# For antihelm B-field
def oneInt(A,B):
    return 4 * ellipe(2*B/(A+B)) / ( (A-B)*sqrt(A+B) )
    
# B-field
def B_field(self,r,t,kk=None,pp=None,Bswitch=None):
    
    """
        For 'cycloid' and 'rotating', test with `test.p.gen_time_steps(t_i=0,t_f=10,step_num=100)`
    """
    
    if kk==None:
        kk = self.k
    if pp==None:
        pp = self.p
    if Bswitch==None:
        Bswitch=1
    
    if pp.B_field_mode == 'ideal':
        
        B_axial_grad = Bswitch*pp.B_axial_grad_reversing + pp.B_axial_grad_nonreversing
        
        Bx = (B_axial_grad + pp.B_trans)*r[0] + pp.B1*r[1] + pp.B2*r[2]
        By = pp.B1*r[0] + (B_axial_grad - pp.B_trans)*r[1] + pp.B3*r[2]
        Bz = pp.B2*r[0] + pp.B3*r[1] - 2*B_axial_grad*r[2]
        
        B_t = array([Bx,By,Bz]) + pp.B0 * Bswitch + pp.B0_nr
    
    elif pp.B_field_mode == 'cycloid':
        B_t = array([0,0,100]) / kk.fund_e * kk.XF_m # cycloid debug (1e-20 is too large, 1e-30 is effectively zero)
    
    elif pp.B_field_mode == 'rotating':
        B_t = array([0,0,0]) / kk.fund_e # rotating debug

    elif pp.B_field_mode == 'antihelm':
        """
            For a single coil of radius R0 with CCW current I at origin, and point P at (x0,0,z0),
            B-field is given as (see Mathematica):
                Bx = C*z0*cosInt
                By = C*z0*sinInt = 0
                Bz = C*(R*oneInt - x0*cosInt)
                where
                    cosInt = -4 * ( A*ellipe(2*B/(A+B)) + (-A+B)*ellipk(2*B/(A+B)) ) / (A-B)*B*sqrt(A+B)
                    sinInt = 0
                    oneInt = 4 * ellipe(2*B/(A+B)) / (A-B)*sqrt(A+B)
                    A = R**2 + x0**2 + z0**2
                    B = -2*R*x0
                    C = mu0*I*R / (4*pi)
        """

        A_down = pp.antihelm_R**2 + sum((r + array([0,0,pp.antihelm_d/2]))**2)
        A_up   = pp.antihelm_R**2 + sum((array([0,0,pp.antihelm_d/2] - r))**2)

        B = -2 * pp.antihelm_R * sqrt(sum(r[0:2]**2))

        C = kk.fund_mu0 * pp.antihelm_I * pp.antihelm_N * pp.antihelm_R / (4*pi) * Bswitch

        Br_down = C * ( r[2] + pp.antihelm_d/2 ) * cosInt(A_down,B)
        Br_up   = C * ( pp.antihelm_d/2 - r[2] ) * cosInt(A_up,B)

        Bz_down = C * (pp.antihelm_R * oneInt(A_down,B) - sqrt(sum(r[0:2]**2)) * cosInt(A_down,B))
        Bz_up   = -C * (pp.antihelm_R * oneInt(A_up,B) - sqrt(sum(r[0:2]**2)) * cosInt(A_up,B))

        B_axgrad_nonreversing = pp.B_axial_grad_nonreversing * array([1,1,-2]) * r
        
        Bx = (Br_down + Br_up) * notNAN(r[0] , sqrt(r[0]**2 + r[1]**2)) + B_axgrad_nonreversing[0]
        By = (Br_down + Br_up) * notNAN(r[1] , sqrt(r[0]**2 + r[1]**2)) + B_axgrad_nonreversing[1]
        Bz = Bz_down + Bz_up + B_axgrad_nonreversing[2]

        B_t = array([Bx,By,Bz]) + pp.B0 * Bswitch + pp.B0_nr
        
    return B_t


##---- Vector operations ----##

# Rotate vector to rotating frame
def to_rot(from_lab,theta,phi):
    """
        Passive rotation. Rotates z-axis of ``from_lab`` frame by (theta,phi) w.r.t. old frame
        
        Defining z-axis of molecule frame to point in quantization direction (E_rot)
        A theta=pi/2 rotation gives x-axis of molecule frame to point in -Z direction.
        So to rotate lab frame vector into molecule frame, (theta=π/2,phi=omega_rot*t-π/2)
    """
    R = array([[cos(theta)*cos(phi), cos(theta)*sin(phi), -sin(theta)],
               [-sin(phi), cos(phi), 0],
               [sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta)]]);
    
    return dot(R,from_lab)

# Magnitude of an n-dim array
def mag(array):
    return sqrt(sum([each**2 for each in array]))

# Inner product of spherical and cartesian rank 1 tensors
def inner_sph_cart(sph,cart):
    """
        inner(sph,cart), where `sph` is (+1,0,-1), and `cart` is (x,y,z).
        
        Definitions in B&C eq 1.26. Can also use dot product defined in eq 1.62, or 5.157 for p in lab frame and q in molecule frame.
    """
    Qz = sph[1]*cart[2]
    Qx = -1/sqrt(2) * (sph[0]-sph[2]) * cart[0]
    Qy = 1j/sqrt(2) * (sph[0]+sph[2]) * cart[1]
    return Qx+Qy+Qz


##---- Special functions ----##

# Zero function
def ling(x):
    return 0

# min(|a|/|b|,1)
def comparator(a,b):
    """
    Returns |a|/|b| if |a|<|b|, else return 1.
    """
    if abs(a)<abs(b):
        return abs(a)/abs(b)
    else:
        return 1

# For antihelm
def notNAN(a,b):
    """
        Give fraction a/b if b≠0, else give 0
    """
    if a==0 and b==0:
        return 1
    elif a!=0 and b==0:
        return float('NAN')
    else:
        return a/b

# Differentiation
def differentiate(x,f):
    return x[:-1]+(x[1]-x[0])/2 , diff(f)/(x[1]-x[0])

# Kronecker delta
def delta(a,b):
    if a==b:
        return 1
    else:
        return 0

# Check if non-zero
def not_zero(a):
    if a==0:
        return 0
    else:
        return 1

# Get rid of machine-precision zeroes, threshold at 10^-16
def ridZero(ob):
    return array([[eacheach if abs(eacheach)>1e-16 else 0 for eacheach in each] for each in ob])


##---- Fitting functions ----##
    
# For fitting purposes
def fit_sine(t,A,omega,phi,C):
    """
        For fitting sine functions.
        
        Usage: popt,pcov = curve_fit(fit_sine, user_time, user_x[, list of initial guess values])
               plot(user_time, fit_sine(user_time,*popt))
    """
    return A*sin(omega*t + phi) + C


##---- Debugging functions ----##

# List all properties of obj
def prop_list(obj):
    for property, value in vars(obj).items():
        print(property, ": ", value)

# Print function code
def print_func(f):
    print(inspect.getsource(f))
    
    
##---- Analysis functions ----##
    
# For microwave studies, with QuTip
def StarkEig_QT(sim,E,B=None,Rswitch=1):
    """
    Solve energy eigenequation to give eigenenergies and eigenvectors.
    Returns (numpy array of eigenvalues , numpy array of Qobj eigenvectors)
    """
    # Redefine E_rot
    if type(E) is not ndarray:
        E = array([0,0,E])
    
    if all(B)==None:
        B = array([0,0,0])

    # Total Hamiltonian
    sim.q.H = ( sim.q.H_hf 
                + sim.q.H_Omega_doubling 
                + sim.q.H_Rot(Rswitch,kk=sim.k,pp=sim.p)
                + sim.q.H_Stark(E,kk=sim.k,pp=sim.p) 
                + sim.q.H_Zeeman(B,kk=sim.k,pp=sim.p) 
                + sim.q.offset )
    
    # Convert to QuTip object
    sim.q.H_QT = Qobj(sim.q.H)
    
    # Solve eigensystem
    sim.q.eigenval_QT , sim.q.eigenvec_QT = sim.q.H_QT.eigenstates()
    
# Load object
def loadObj(filename,path=None):
    
    if filename[-4:] != '.obj':
        filename = filename + '.obj'
    
    if path!=None:
        filename = path+filename
    
    try:
        print('Loading from '+filename)
        with gzip.open(filename, 'rb') as f_handle:
            return pickle.load(f_handle)
    except:
        try:
            print('Loading from '+filename)
            with open(filename, 'rb') as f_handle:
                return pickle.load(f_handle)
        except:
            print(filename+' not found ><')

# Load data.csv
def loadData(path):
    return loadtxt(open(path+"data.csv"), delimiter=",", skiprows=1)

# Equal check
def eq_check(a,b):
    return a in b

# Unequal check
def neq_check(a,b):
    return a not in b

# Find files
def find_files(kw_list=None,eq_list=None,path=None):
    """
        Find all files in the current directory (or `path`) which satisfy conditions:
            - `kw eq filename`, where kw is a search keyword, and eq is boolean for `in` (True) or `not in` (False).
            - Iterate for each keyword in `kw_list` for corresponding condition in `eq_list`.
            
        E.g. In a list of ['I','need','a','pay','raise'], `find_files(kw_list=['a'],eq_list=[False])` returns ['I','need']
    """
    
    # List all files in current folder
    listAll = array(os.listdir(path))
    
    if kw_list==None: # return all files
        return listAll.tolist()
    else:
        # Determine check function for each keyword
        if eq_list==None:
            check = [ eq_check for n in range(len(kw_list)) ]
        else:
            check = [ eq_check if each else neq_check for each in eq_list ]

        return listAll[ [ all([ check[n](kw_list[n],each) for n in range(len(kw_list)) ]) for each in listAll ] ].tolist()


##---- Random number generators ----##

# 1D RNG for a user defined PDF
class arb_dist:
    def __init__(self,xs,Ps):
        """
            Takes in an arbitrary probability distribution function P(x) and generates random numbers based on this distribution.
            
            Inputs: Ps (values of PDF) at various values of xs.
            Output: An object.
            
            By "Inverse Transform Sampling":
            - if X is a continuous random variable,
            - with CDF F_X, F_X(x) := P(X≤x),
            - then the random variable Y = F_X(X) has a uniform distribution on [0,1].
            
            Proof:
            - suppose F_Y is a CDF of a U(0,1) random variable y,
            - F_Y(y) = y
                     = F_X(F_X_inv(y))
                     = P(X ≤ F_X_inv(y))
                     = P(F_X(X) ≤ y)
            - but F_Y(y) = P(Y ≤ y), so Y = F_X(X) and X = F_X_inv(Y).
        """
        if any(Ps<0):
            print('Ps is less than zero.')
            return
        
        if len(xs) != len(Ps):
            print('Lengths of xs and Ps are not the same.')
            return
            
        self.xs = xs
        self.Ps = Ps
        
        self.Ps_norm = self.Ps/sum(self.Ps)
        self.Ps_CDF = self.Ps_norm.cumsum()
        
        # Extrapolating all the way to 0
        if self.Ps_CDF[0] != 0:
            self.Ps_CDF = concatenate((array([0]),self.Ps_CDF))
            self.xs = concatenate((array([self.xs[0]-(mean(diff(self.xs)))]),self.xs))
        
        self.inv_CDF = interp1d(self.Ps_CDF,self.xs)
        
    def random(self,N=None):
        return self.inv_CDF(random(N)).flatten()
            
# 2D RNG for a user defined PDF
class arb_dist_2D:
    def __init__(self,xs,ys,Ps):
        """
            Takes in an arbitrary probability distribution function P(x,y) and generates random numbers based on this distribution.
            
            Inputs: Ps (values of PDF) at various values of xs,ys.
            Output: An object.
            
            Similar to how we generate a 1D RNG with arbitrary PDF.
            1. First integrate out one axis (x) to give a PDF of one variable: P(y), and use the 1D inverse CDF method to get a random y.
            2. Using the random y, obtain the PDF at that y as a function of x: P(x|y), and use the same 1D inverse CDF method to get a random x.
        """
        if (size(xs),size(ys)) != shape(Ps):
            print('Shape of Ps != (size of xs , size of ys)')
            return
            
        self.xs,self.ys,self.Ps = xs,ys,Ps
        
        # Integrating xs out
        self.Ps_y = sum(self.Ps,axis=0)
        self.Ps_norm_y = self.Ps_y/sum(self.Ps_y)
        self.Ps_CDF_y = self.Ps_norm_y.cumsum()
        
        # Extrapolating all the way to 0
        if self.Ps_CDF_y[0] != 0:
            self.Ps_CDF_y = concatenate((array([0]),self.Ps_CDF_y))
            self.ys = concatenate((array([self.ys[0]-(mean(diff(self.ys)))]),self.ys))
            
        self.inv_CDF_y = interp1d(self.Ps_CDF_y,self.ys)
        
        self.Ps_norm_interp = interp2d(self.xs,ys,self.Ps.T)
        
    def random(self,N=None):
        
        self.rand_ys = self.inv_CDF_y(random(N)).flatten()
        self.rand_xs = []
        
        for y in self.rand_ys:
            Ps_x = self.Ps_norm_interp(self.xs,y)
            Ps_norm_x = Ps_x/sum(Ps_x)
            Ps_CDF_x = Ps_norm_x.cumsum()
            
            # Extrapolating all the way to 0
            if Ps_CDF_x[0] != 0:
                Ps_CDF_x = concatenate((array([0]),Ps_CDF_x))
                xs = concatenate((array([self.xs[0]-(mean(diff(self.xs)))]),self.xs))
                
            inv_CDF_x = interp1d(Ps_CDF_x,xs)
            
            self.rand_xs.append(inv_CDF_x(random()))
            
        self.rand_xs = array(self.rand_xs).flatten()
        
        return array([self.rand_xs,self.rand_ys]).T


##---- Other useful classes and functions----##


### Bloch Vector and Rotation

# Tool to visualize a series of rotations of the Bloch vector on a Bloch sphere
class BlochVector:
    def __init__(self,start_state=None):
        """
            start_state takes in a 1x3 array, initializes to array([0,0,1]) if None.
            
            Typical usage:
                blah.clear() # clears any previous journeys
                blah.move(pi/4,pi/2,pi/2) # first rotation
                blah.move(0,0,pi)         # second rotataion
                blah.move(pi/4,pi/2,pi/2) # third rotation
                blah.show() # plot journey on Bloch sphere
        """
        self.start = array([0,0,1]) if start_state==None else start_state # starting state
        self.journey = [] # log of all rotation paths
        self.current= self.start # current state

    def Rx(self,x):
        """
            Active rotation about x-axis for angle x
        """
        return array([[1,0,0],[0,cos(x),-sin(x)],[0,sin(x),cos(x)]])

    def Ry(self,x):
        """
            Active rotation about y-axis for angle x
        """
        return array([[cos(x),0,sin(x)],[0,1,0],[-sin(x),0,cos(x)]])

    def Rz(self,x):
        """
            Active rotation about z-axis for angle x
        """
        return array([[cos(x),-sin(x),0],[sin(x),cos(x),0],[0,0,1]])

    def R_arby(self,a,x):
        """
            Active rotation about some axis along y-z plane with angle a from z-axis for angle x
        """
        return self.Rx(-a)@self.Rz(x)@self.Rx(a)

    def R_arb(self,a,b,x):
        """
            Active rotation about some axis (theta=a,phi=b) for angle x
        """
        return self.Rz(-(pi/2-b))@self.R_arby(a,x)@self.Rz(pi/2-b)

    def move(self,a,b,x):
        """
            Rotates about rotation vector (theta=a,phi=b) for angle x
        """
        self.journey.append(array([ self.R_arb(a,b,x/10*i)@self.current for i in range(10) ]).T)
        self.current = self.R_arb(a,b,x)@self.current
        
    def show(self):
        self.b = Bloch() # Initialize Bloch sphere from QuTip
        self.b.add_vectors(self.start) # add starting vector
        self.b.add_vectors(self.current) # add current vector
        for each in self.journey: # plot journey out as points on Bloch sphere
            self.b.add_points(each)
        self.b.show() # plot it out
        
    def clear(self):
        self.journey = [] # reset journey
        self.current = self.start # reset current state


### Franck-Condon
"""
    There is a closed-form solution for Morse potential in `J. Chem. Phys. 88 (7): 4535 (1988)`, but there are a lot of extreme numbers (e.g. small^large or large/large!) which makes numerical evaluation of the function tricky.
    
    Approximating the diatomic potentials as harmonic potentials and treating the harmonicities of both electronic states of interest to be close to each other yields and approximate analytic expression as stated in `J. Chem. Phys. 74, 6980 (1981)`. This should work for low v numbers if anharmonicity is not too crazy.
"""

# Tool for Franck-Condon factors
class elec:
    def __init__(self,name,Be,we,wexe,Te,De=None,ae=None):
        """
            From Spectra of Atoms and Molecules (Bernath) p.212, Eq 7.27-7.33:
                
                For a Morse potential of form V(r) = D( 1 - exp( -beta*(r-re) ) )^2, eigenenergy is given by:
                
                    E [cm^-1] = we(v+1/2) - wexe(v+1/2)^2 + Be J(J+1) - De [J(J+1)]^2 - ae(v+1/2)[J(J+1)],
                    
                    we = beta * √( 100 D h / (2 π^2 c µ) )
                    wexe = 100 h beta^2 / ( 8 π^2 µ re^2 c )
                    Be = h / ( 800 π^2 µ re^2 c )
                    De = 4 Be^3 / we^2 ; Kratzer relationship, applies to all realistic diatomic potentials
                    ae = [ 6 √( wexe Be^3 ) - 6 Be^2 ] / we ; Pekeris relationship, applies only to Morse potential
                    
                    {beta,D,all variables on LHS} in [cm^-1], {h,c,µ(reduced mass)} in S.I.
        """
        h,amu,c = 6.63e-34,1.66e-27,299792458
        mu = 232*19/251*amu
        
        self.name = name # electronic state designation
        self.Be = Be # rotation constant [cm^-1]
        self.re = sqrt(h/100/(8*pi**2*mu*c)/Be) # internuclear distance at potential minimum [m]
        self.we = we # harmonic constant [cm^-1]
        self.wexe = wexe # anharmonic constant [cm^-1]
        self.Te = Te # energy of potential minimum w.r.t. zero point
        
        self.De = 0 if De==None else De
        self.ae = 0 if ae==None else ae
        
    def G(self,v):
        """
            G(v) = we(v+1/2) - wexe(v+1/2)^2 ; vibrational energy
        """
        return self.we*(v+1/2) - self.wexe*(v+1/2)**2

    def E(self,v,J):
        """
            E [cm^-1] = we(v+1/2) - wexe(v+1/2)^2 + Be J(J+1) - De [J(J+1)]^2 - ae(v+1/2)[J(J+1)]
        """
        return self.G(v) + self.Be*J*(J+1) - self.De*(J*(J+1))**2 - self.ae*(v+1/2)*J*(J+1) + self.Te

# From Dan's paper
states = { '3D1': elec('3∆1', 0.24311, 656.96 , 1.920, 0       , ae=1.00e-3),
           '1S+': elec('1∑+', 0.24601, 657.90 , 2.26 , 314.0   , ae=1.12e-3),
           'O0+': elec('Ω0+', 0.23583, 626.67 , 1.88 , 10487.1 , ae=0.97e-3),
           'O0-': elec('Ω0-', 0.22947, 593.467, 1.822, 14620.81, ae=0.95e-3) }

# Calculate Franck-Condon factor
def qv1v2(s1,s2,v1,v2):
    """
        From Spectra of Atoms amd Molecules (Bernath), p.363-364, Q9, referenced to Nicholls, J. Chem. Phys. 74, 6980 (1981)
        
        Appoximate Franck-Condon factors derived assuming harmonic potentials with an averaged vibrational frequency:
        
            √(we_bar) = 2*√(we1*we2) / (√(we1) + √(we2))
            
        which works only when we1 and we2 are close to each other.
            
        Approximate analytic expressions only given up to min(v1,v2)=3.
    """
    h,amu,c = 6.63e-34,1.66e-27,299792458
    mu = 232*19/251*amu

    # From Bernath,
    Delta_re = (s1.re - s2.re) * 1e10
    we_bar = ( 2*sqrt(s1.we*s2.we) / (sqrt(s1.we) + sqrt(s2.we)) )**2
    u = Delta_re**2 * mu/amu * we_bar / 67.44

    vm,vM = min(v1,v2),max(v1,v2)

    # From Nicholls,
    if vm==0:
        return u**vM * exp(-u) / factorial(vM)
    elif vm==1:
        return u**(vM-1) * exp(-u) / factorial(vM) * (u-vM)**2
    elif vm==2:
        return u**(vM-2) * exp(-u) / factorial(2) / factorial(vM) * ( (u-vM)**2 - vM )**2
    elif vm==3:
        return u**(vM-3) * exp(-u) / factorial(3) / factorial(vM) * ( (u-vM)**3 - vM*( 3*(u-vM) + 2) )**2
    else:
        return -1

# Generate Franck-Condon table
def FCtable(s1,s2,fig_return=None):
    """
        Generates Franck-Condon table up to v',v"=3 with analytic approximation from Bernath/Nicholls.
    """

    fcs = [ [ qv1v2(s1,s2,v1,v2) for v2 in range(4) ] for v1 in range(4) ]

    # Determine which is of lower energy, takes advantage of symmetry in FC
    sa = s1 if s1.Te <= s2.Te else s2
    sb = s2 if s1.Te <= s2.Te else s1

    fig=figure()
    im = imshow(fcs)
    xlabel("v' ("+sb.name+')')
    ylabel('v" ('+sa.name+')')
    xticks([ x for x in range(len(fcs[0])) ])
    yticks([ x for x in range(len(fcs)) ])

    for i in range(len(fcs)):
        for j in range(len(fcs[0])):
            if all(2*fcs[i][j] > max(fcs)):
                yanse = 'black'
            else:
                yanse = 'white'
            annotate('{:.3f}'.format(fcs[i][j]),(i,j),ha='center',va='center',color=yanse)

    tight_layout()
    colorbar(im)
    show()
    
    if fig_return==True:
        return fig


### Hönl-London

# Calculate Hönl-London factors
def HL(OO,O,JJ,J):
    """
        Calculates Hönl-London factors for a transition from lower {Ω,J}={OO,JJ} to upper {Ω,J}={O,J}.
        
        Table 9.4 p.335 from Spectra of Atoms and Molecules (Bernath)
    """
    if abs(JJ-J)>1: # only for PQR transitions
        return 0
    elif JJ<OO or J<O: # There should be no J<Ω
        return 0
    else:
        if O==OO:
            if J-JJ==1: # R
                return (JJ+1+OO) * (JJ+1-OO) / (JJ+1)
            elif J==JJ: # Q
                return (2*JJ+1) * OO**2 / (JJ * (JJ+1))
            elif J-JJ==-1: # P
                return (JJ+OO) * (JJ-OO) / JJ
        elif O-OO==1:
            if J-JJ==1: # R
                return (JJ+2+OO) * (JJ+1+OO) / 2 / (JJ+1)
            elif J==JJ: # Q
                return (JJ+1+OO) * (JJ-OO) * (2*JJ+1) / 2 / (JJ * (JJ+1))
            elif J-JJ==-1: # P
                return (JJ-1-OO) * (JJ-OO) / 2 / JJ
        elif O-OO==-1:
            if J-JJ==1: # R
                return (JJ+2-OO) * (JJ+1-OO) / 2 / (JJ+1)
            elif J==JJ: # Q
                return (JJ+1-OO) * (JJ+OO) * (2*JJ+1) / 2 / (JJ * (JJ+1))
            elif J-JJ==-1: # P
                return (JJ-1+OO) * (JJ+OO) / 2 / JJ


### Optical pumping

def Op_Pump(SheepP2_bool=None,MonkeyP1_bool=None,v1_30GHz_bool=None,GoatQ1_bool=None,GoatP1_bool=None,DogP1_bool=None,tf=None,n0=None,result_bool=None):
    """
        If bool==None, set to be False (no pump).
        tf is in seconds.
        n0 is in n01D,n01B,n02,n11,n12,n21,n22,nO
    """
    
    # FC and HL
    q00,q01,q02 = 0.576,0.318,0.087
    HL11,HL12,HL01,HL02 = 0.5,0.5,1,0

    # "Rabi rate"
    A = 150 # to give a SheepP2 decay time constant of about 9ms

    # Switch
    if SheepP2_bool==None:
        SheepP2_bool = False
    if MonkeyP1_bool==None:
        MonkeyP1_bool = False
    if v1_30GHz_bool==None:
        v1_30GHz_bool = False
    if DogP1_bool==None:
        DogP1_bool = False
    if GoatQ1_bool==None:
        GoatQ1_bool = False
    if GoatP1_bool==None:
        GoatP1_bool = False
    if tf==None:
        tf = 0.15

    ### Rates ###

    """n01D,n01B,n02,n11,n12,n21,n22,nO"""
    biao = [r'{v=0,J=1}, D',r'{v=0,J=1}, B',r'{v=0,J=2}',r'{v=1,J=1}',r'{v=1,J=2}',r'{v=2,J=1}',r'{v=2,J=2}',r'Elsewhere']
    if any(n0==None):
        n0 = array([0,0,1,0,0,0,0,0])

    # Sheep P(2)
    """
        Sheep P(2)

        d(n01D)/dt = n02 * q00 * HL11 * 1/6
        d(n01B)/dt = n02 * q00 * HL11 * 5/6
        d(n02)/dt  = - n02 * ( 1 - q00 * HL12 )
        d(n11)/dt  = n02 * q01 * HL11
        d(n12)/dt  = n02 * q01 * HL12
        d(nO)/dt   = n02 * ( 1 - q00 - q01 )
    """
    Gamma_SheepP2 = array([[ 0 , 0 , A * q00 * HL11 * 1/6 , 0 , 0 , 0 , 0 , 0 ],
                           [ 0 , 0 , A * q00 * HL11 * 5/6 , 0 , 0 , 0 , 0 , 0 ],
                           [ 0 , 0 , - A * (1 - q00 * HL12) , 0 , 0 , 0 , 0 , 0 ],
                           [ 0 , 0 , A * q01 * HL11 , 0 , 0 , 0 , 0 , 0 ],
                           [ 0 , 0 , A * q01 * HL12 , 0 , 0 , 0 , 0 , 0 ],
                           [ 0 , 0 , A * q02 * HL11 , 0 , 0 , 0 , 0 , 0 ],
                           [ 0 , 0 , A * q02 * HL12 , 0 , 0 , 0 , 0 , 0 ],
                           [ 0 , 0 , A * (1 - q00 - q01 - q02) , 0 , 0 , 0 , 0 , 0 ]])

    # Goat Q(1)
    """
        Goat Q(1)

        d(n01D)/dt = n01B * q00 * HL11 * 1/6
        d(n01B)/dt = - n01B * (1 - q00 * HL11 * 5/6 )
        d(n02)/dt  = n01B * q00 * HL12
        d(n11)/dt  = n01B * q01 * HL11
        d(n12)/dt  = n01B * q01 * HL12
        d(nO)/dt   = n01B * ( 1 - q00 - q01 )
    """
    Gamma_GoatQ1 = array([[ 0 , A * q00 * HL11 * 1/6 , 0 , 0 , 0 , 0 , 0 , 0 ],
                          [ 0 , - A * (1 - q00 * HL11 * 5/6) , 0 , 0 , 0 , 0 , 0 , 0 ],
                          [ 0 , A * q00 * HL12 , 0 , 0 , 0 , 0 , 0 , 0 ],
                          [ 0 , A * q01 * HL11 , 0 , 0 , 0 , 0 , 0 , 0 ],
                          [ 0 , A * q01 * HL12 , 0 , 0 , 0 , 0 , 0 , 0 ],
                          [ 0 , A * q02 * HL11 , 0 , 0 , 0 , 0 , 0 , 0 ],
                          [ 0 , A * q02 * HL12 , 0 , 0 , 0 , 0 , 0 , 0 ],
                          [ 0 , A * (1 - q00 - q01 - q02) , 0 , 0 , 0 , 0 , 0 , 0 ]])

    # Goat P(1)
    """
        Goat P(1)

        d(n01D)/dt = n01B * q00 * HL01 * 1/6
        d(n01B)/dt = - n01B * (1 - q00 * HL01 * 5/6 )
        d(n02)/dt  = n01B * q00 * HL02
        d(n11)/dt  = n01B * q01 * HL01
        d(n12)/dt  = n01B * q01 * HL02
        d(nO)/dt   = n01B * ( 1 - q00 - q01 )
    """
    Gamma_GoatP1 = array([[ 0 , A * q00 * HL01 * 1/6 , 0 , 0 , 0 , 0 , 0 , 0 ],
                          [ 0 , - A * (1 - q00 * HL01 * 5/6) , 0 , 0 , 0 , 0 , 0 , 0 ],
                          [ 0 , A * q00 * HL02 , 0 , 0 , 0 , 0 , 0 , 0 ],
                          [ 0 , A * q01 * HL01 , 0 , 0 , 0 , 0 , 0 , 0 ],
                          [ 0 , A * q01 * HL02 , 0 , 0 , 0 , 0 , 0 , 0 ],
                          [ 0 , A * q02 * HL01 , 0 , 0 , 0 , 0 , 0 , 0 ],
                          [ 0 , A * q02 * HL02 , 0 , 0 , 0 , 0 , 0 , 0 ],
                          [ 0 , A * (1 - q00 - q01 - q02) , 0 , 0 , 0 , 0 , 0 , 0 ]])
    
    # Monkey P(1)
    """
        Monkey P(1)

        d(n01D)/dt = n11 * q00 * HL01 * 1/6
        d(n01B)/dt = n11 * q00 * HL01 * 5/6
        d(n02)/dt  = n11 * q00 * HL02
        d(n11)/dt  = - n11 * (1 - q01 * HL01)
        d(n12)/dt  = n11 * q01 * HL02
        d(nO)/dt   = n11 * ( 1 - q00 - q01 )
    """
    Gamma_MonkeyP1 = array([[ 0 , 0 , 0 , A * q00 * HL01 * 1/6 , 0 , 0 , 0 , 0 ],
                            [ 0 , 0 , 0 , A * q00 * HL01 * 5/6 , 0 , 0 , 0 , 0 ],
                            [ 0 , 0 , 0 , A * q00 * HL02 , 0 , 0 , 0 , 0 ],
                            [ 0 , 0 , 0 , - A * (1 - q01 * HL01) , 0 , 0 , 0 , 0 ],
                            [ 0 , 0 , 0 , A * q01 * HL02 , 0 , 0 , 0 , 0 ],
                            [ 0 , 0 , 0 , A * q02 * HL01 , 0 , 0 , 0 , 0 ],
                            [ 0 , 0 , 0 , A * q02 * HL02 , 0 , 0 , 0 , 0 ],
                            [ 0 , 0 , 0 , A * (1 - q00 - q01 - q02) , 0 , 0 , 0 , 0 ]])

    # v=1 30GHz
    """
        v=1 30GHz

        d(n01D)/dt = n11 * q00 * HL01 * 1/6
        d(n01B)/dt = n11 * q00 * HL01 * 5/6
        d(n02)/dt  = n11 * q00 * HL02
        d(n11)/dt  = - n11 * (1 - q01 * HL01)
        d(n12)/dt  = n11 * q01 * HL02
        d(nO)/dt   = n11 * ( 1 - q00 - q01 )
    """
    Gamma_v1_30GHz = array([[ 0 , 0 , 0 , 0 , A * q00 * HL01 * 1/6 , 0 , 0 , 0 ],
                            [ 0 , 0 , 0 , 0 , A * q00 * HL01 * 5/6 , 0 , 0 , 0 ],
                            [ 0 , 0 , 0 , 0 , A * q00 * HL02 , 0 , 0 , 0 ],
                            [ 0 , 0 , 0 , 0 , A * q01 * HL01 , 0 , 0 , 0 ],
                            [ 0 , 0 , 0 , 0 , - A * (1 - q01 * HL02) , 0 , 0 , 0 ],
                            [ 0 , 0 , 0 , 0 , A * q02 * HL01 , 0 , 0 , 0 ],
                            [ 0 , 0 , 0 , 0 , A * q02 * HL02 , 0 , 0 , 0 ],
                            [ 0 , 0 , 0 , 0 , A * (1 - q00 - q01 - q02) , 0 , 0 , 0 ]])
    
    # Dog P(1)
    """
        Monkey P(1)

        d(n01D)/dt = n21 * q00 * HL01 * 1/6
        d(n01B)/dt = n21 * q00 * HL01 * 5/6
        d(n02)/dt  = n21 * q00 * HL02
        d(n11)/dt  = n21 * q01 * HL01)
        d(n12)/dt  = n21 * q01 * HL02
        d(n21)/dt  = - n21 * (1 - q01 * HL01)
        d(n22)/dt  = n21 * q01 * HL02
        d(nO)/dt   = n21 * ( 1 - q00 - q01 )
    """
    Gamma_DogP1 = array([[ 0 , 0 , 0 , 0 , 0 , A * q00 * HL01 * 1/6 , 0 , 0 ],
                            [ 0 , 0 , 0 , 0 , 0 , A * q00 * HL01 * 5/6 , 0 , 0 ],
                            [ 0 , 0 , 0 , 0 , 0 , A * q00 * HL02 , 0 , 0 ],
                            [ 0 , 0 , 0 , 0 , 0 , A * q01 * HL01 , 0 , 0 ],
                            [ 0 , 0 , 0 , 0 , 0 , A * q01 * HL02 , 0 , 0 ],
                            [ 0 , 0 , 0 , 0 , 0 , - A * (1 - q02 * HL01) , 0 , 0 ],
                            [ 0 , 0 , 0 , 0 , 0 , A * q02 * HL02 , 0 , 0 ],
                            [ 0 , 0 , 0 , 0 , 0 , A * (1 - q00 - q01 - q02) , 0 , 0 ]])


    # Total Rate
    Gamma = SheepP2_bool*Gamma_SheepP2 + MonkeyP1_bool*Gamma_MonkeyP1 + v1_30GHz_bool*Gamma_v1_30GHz + DogP1_bool*Gamma_DogP1 + GoatQ1_bool*Gamma_GoatQ1 + GoatP1_bool*Gamma_GoatP1


    ### Plot ###

    ts = linspace(0,tf) # [s]

    ns = array([ expm(Gamma*t)@n0 for t in ts ]).T

    figure(figsize=(8,4))
    for each in ns:
        plot(1e3*ts,each/sum(ns[:,-1]))
    plot(1e3*ts,sum(ns/sum(ns[:,-1]),axis=0),'--')
    ylabel('Intensity (arb. unit)')
    xlabel('Time (ms)')
    grid()
    legend(biao)
    title('With '+SheepP2_bool*'Sheep_P(2) '+MonkeyP1_bool*'Monkey_P(1) '+v1_30GHz_bool*'30GHz '+DogP1_bool*'Dog_P(1) '+GoatQ1_bool*'Goat_Q(1) '+GoatP1_bool*'Goat_P(1) ')
    tight_layout()
    show()

    print('Population at the start:')
    for n in range(len(n0)):
        print('\t{}\t{:.3f}'.format(biao[n],n0[n]/sum(ns[:,-1])))
    print('\t'+21*'-')
    print('\t{v=0,J=1}, D+B'+'\t{:.3f}\n'.format((n0[0]+n0[1])/sum(ns[:,-1])))
    
    print('Population at late times:')
    for n in range(len(n0)):
        print('\t{}\t{:.3f}'.format(biao[n],ns[n,-1]/sum(ns[:,-1])))
    print('\t'+21*'-')
    print('\t{v=0,J=1}, D+B'+'\t{:.3f}'.format((ns[0,-1]+ns[1,-1])/sum(ns[:,-1])))
        
    if result_bool==True:
        return ts,ns


### Mathieu parameter plot

# Mathieu parameters

"""
    For Gen. I setup:
        Z0 = 0.130m
        R0 = 0.05531m
"""

def a0(U0,m=251,Omega=2*pi*50e3,Z0=0.130):
    return -4 * 1.6e-19 * U0 / (m * 1.66e-27 * Omega**2 * Z0**2)

def ap(Vp,m=251,Omega=2*pi*50e3,R0=0.05531):
    return 8 * 1.6e-19 * Vp / (m * 1.66e-27 * Omega**2 * R0**2)

def q0(Vrfpp,m=251,Omega=2*pi*50e3,R0=0.05531):
    return 2 * 1.6e-19 * Vrfpp / (m * 1.66e-27 * Omega**2 * R0**2)

def fs(Vrfpp,U0,Vp,m=251,f_rf=50e3,Z0=0.130,R0=0.05531): # f_rf in kHz here
    
    fx = f_rf/2 * sqrt( -abs( a0(U0,m=m,Omega=2*pi*f_rf,Z0=Z0) ) + ap(Vp,m=m,Omega=2*pi*f_rf,R0=R0) + q0(Vrfpp,m=m,Omega=2*pi*f_rf,R0=R0)**2/2 )
    fy = f_rf/2 * sqrt( -abs( a0(U0,m=m,Omega=2*pi*f_rf,Z0=Z0) ) - ap(Vp,m=m,Omega=2*pi*f_rf,R0=R0) + q0(Vrfpp,m=m,Omega=2*pi*f_rf,R0=R0)**2/2 )
    fz = f_rf/2 * sqrt( 2*abs( a0(U0,m=m,Omega=2*pi*f_rf,Z0=Z0) ) )

    return array([ fx,fy,fz ])

def show_Mathieu(Vrfpp,U0,Vp,f_rf=50e3,Z0=0.130,R0=0.05531,mode=None,return_bool=None,qlim=None,alim=None,contour_lines=None): # f_rf in kHz here
    """
        mode = None for (q,a), 'ap' for (q,a')
        Fit end-cap potential to the form U0(z^2/Z0^2) to get Z0.
        Fit RF potential to the form V0(x^2-y^2)/R0^2 to get R0.
    """
    
    qs = linspace(0,1,600)
    aS = linspace(-0.3,0.3,600)
    Qs,As = meshgrid(qs,aS)
    ms = [251, 251-19, 251+19]
    
    if mode==None:
        """
            Shows x and y trapping as separate points in the stability diagram.
        """
        zorders = [30,20,10]
        species_c = ['tab:orange','tab:green','tab:red']
        fig = figure(figsize=(8,5))
        bound1 = mathieu_a(0,qs)
        bound2 = mathieu_b(1,qs)
        for n in range(3):
            """
                Stability region is bound by ±( A(0,q) - a0 ) from the left and ±( B(1,q) - a0 ) from the right
            """
            plot(qs, bound1,'--',c='tab:blue')
            plot(qs, bound2,'--',c='tab:blue')
            ct = contour(Qs,As,f_rf/2*sqrt(As+Qs**2/2)/1000,cmap='terrain',levels=contour_lines) # Plot trap frequency contour lines
            fill_between(qs,
                         bound1,
                         bound2,
                         facecolor='tab:grey',
                         zorder=0,
                         alpha=0.1)
            if alim==None:
                ylim( [-0.2,0.1] )
            else:
                ylim( alim )
            if qlim==None:
                pass
            else:
                xlim(qlim)
            grid()
            xlabel('$q_0$')
            ylabel('$a$')
            plot(q0(Vrfpp,m=ms[n],Omega=2*pi*f_rf,R0=R0), -abs(a0(U0,m=ms[n],Omega=2*pi*f_rf,Z0=Z0)) + ap(Vp,m=ms[n],Omega=2*pi*f_rf,R0=R0),'.',c=species_c[n],zorder=zorders[n])
            plot(q0(Vrfpp,m=ms[n],Omega=2*pi*f_rf,R0=R0), -abs(a0(U0,m=ms[n],Omega=2*pi*f_rf,Z0=Z0)) - ap(Vp,m=ms[n],Omega=2*pi*f_rf,R0=R0),'.',c=species_c[n],zorder=zorders[n])

        legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=species_c[0], label='ThF$^+$'),
                           Line2D([0], [0], marker='o', color='w', markerfacecolor=species_c[1], label='Th$^+$'),
                           Line2D([0], [0], marker='o', color='w', markerfacecolor=species_c[2], label='ThF$_2^+$')]

        # Create the figure
        legend(handles=legend_elements)
        tight_layout()
        cb = colorbar(ct)
        cb.set_label('kHz')
        fig.tight_layout()
        show()
    elif mode=='ap':
        """
            Shows stability taking both x and y into account.
        """
        species = [ 'ThF$^+$', 'Th$^+$', 'ThF$_2^+$']
        fig,ax = subplots(1,3,figsize=(12,3))
        for n in range(3):
            """
                Stability region is bound by -( A(0,q) - a0 ) from the left and ( B(1,q) - a0 ) from the right.
            """
            stable_region = array([ min(each) for each in array([ mathieu_b(1,qs)-a0(U0,m=ms[n],Omega=2*pi*f_rf,Z0=Z0), - (mathieu_a(0,qs)-a0(U0,m=ms[n],Omega=2*pi*f_rf,Z0=Z0)) ]).T ])
            ax[n].plot(qs, stable_region )
            ax[n].fill_between(qs,0,stable_region,facecolor='tab:grey')
            if alim==None:
                ax[n].set_ylim([0,1.1*max(stable_region)])
            else:
                ax[n].set_ylim( alim )
            if qlim==None:
                pass
            else:
                ax[n].set_xlim(qlim)
            ax[n].grid()
            ax[n].set_xlabel('$q_0$')
            ax[n].set_ylabel('$a\'$')
            ax[n].set_title(species[n])
            ax[n].plot(q0(Vrfpp,m=ms[n],Omega=2*pi*f_rf,R0=R0),ap(Vp,m=ms[n],Omega=2*pi*f_rf,R0=R0),'o')
        fig.tight_layout()
        show()
    
    # Print parameters
    print('\
    \tThF\t\tTh\t\tThF2')
    print('\
    a0\t{:.5f}\t{:.5f}\t{:.5f}\n\
    a\'\t{:.3g}\t\t{:.3g}\t\t{:.3g}\n\
    q0\t{:.3g}\t\t{:.3g}\t\t{:.3g}\n\
    fx\t{:.2f} kHz\t{:.2f} kHz\t{:.2f} kHz\n\
    fy\t{:.2f} kHz\t{:.2f} kHz\t{:.2f} kHz\n\
    fz\t{:.2f} kHz\t{:.2f} kHz\t{:.2f} kHz'.format(a0(U0,m=251,Omega=2*pi*f_rf,Z0=Z0),a0(U0,m=251-19,Omega=2*pi*f_rf,Z0=Z0),a0(U0,m=251+19,Omega=2*pi*f_rf,Z0=Z0),
                                                  ap(Vp,m=251,Omega=2*pi*f_rf,R0=R0),ap(Vp,m=251-19,Omega=2*pi*f_rf,R0=R0),ap(Vp,m=251+19,Omega=2*pi*f_rf,R0=R0),
                                                  q0(Vrfpp,m=251,Omega=2*pi*f_rf,R0=R0),q0(Vrfpp,m=251-19,Omega=2*pi*f_rf,R0=R0),q0(Vrfpp,m=251+19,Omega=2*pi*f_rf,R0=R0),
                                                  *(array([fs(Vrfpp,U0,Vp,m=251,f_rf=f_rf,Z0=Z0,R0=R0),fs(Vrfpp,U0,Vp,m=251-19,f_rf=f_rf,Z0=Z0,R0=R0),fs(Vrfpp,U0,Vp,m=251+19,f_rf=f_rf,Z0=Z0,R0=R0)]).T.flatten())/1000))

    if return_bool==True:
        return fig

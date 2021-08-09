'''
************************ ADDED TO SUPPORT ARC ALGORITHM ************************

Contains functions needed for the ARC policy

Reference:
"Asymptotic Randomised Control with applications to bandits",
S. N. Cohen and T. Treetanthiploet, arXiv:2010.07252, 2020.

'''

from __future__ import absolute_import
from __future__ import division
# Using Type Annotations.
from __future__ import print_function

from enum import Enum
from typing import Optional, Sequence, Text

import numpy as np

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
import tensorflow_probability as tfp
from tf_agents.bandits.policies import constraints
from tf_agents.bandits.policies import linalg
from tf_agents.bandits.specs import utils as bandit_spec_utils
from tf_agents.policies import tf_policy
from tf_agents.policies import utils as policy_utilities
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import policy_step
from tf_agents.typing import types

import scipy.optimize
from scipy.stats import norm
import time

tfd = tfp.distributions


def ARC_policy(self, m, Sigma, rho, beta):
    """
    Function that computes the chosen action according to the ARC algorithm

    References:
    1.  "Asymptotic Randomised Control with applications to bandits",
        S. N. Cohen and T. Treetanthiploet, arXiv:2010.07252, 2020.

    2.  https://github.com/ttreetanthiploet/ARC-with-bandits

    """
    def Gaussian_density(self,z):
      return 1/np.sqrt(2*np.pi)*np.exp(-z**2/2)

    def L_input(self, m, Sigma, Lambda):
      """ Computes input variables for L """
      #Sigma = self.Sigma(d)
      B = self.B
      P = self.P

      #### WRONG - remove later
      d = self.d   # np.diag(S)*10
      ####
      S = np.array(np.matmul(np.matmul(B, Sigma), np.transpose(B)))
      v_d = np.array(1 + np.diag(S) + 1/P)
      b_m = np.array(np.matmul(B,m))
      h = np.ones(len(b_m))#Gaussian_density(self, b_m/np.sqrt(v_d))/np.sqrt(v_d)
      g = np.zeros(len(b_m))#h*b_m/v_d
      S_sq = S**2
      P_over_1d = P/(1+d)                       # Problem here - needs d
      diag_s_p = np.array(np.diag(S) + 1/P)
      coeff_first_term = P_over_1d - 1/diag_s_p
      return (coeff_first_term, S_sq, S, g, h, diag_s_p)

    def L_compute(self, a, Lambda, coeff_first_term, S_sq, S, g, h, diag_s_p):
      """
      Computes L from the input variables

      See Appendix C.4 in ARC paper (Reference 1) and Section 3.2 in Research Summary

      """
      nu_g = nu(self, a, Lambda)*g
      First_term= 1/2*coeff_first_term * np.matmul(nu_g,  S_sq)
      nu_h = nu(self, a, Lambda)*h
      nu_h2 = nu(self, a, Lambda)*h**2
      second_term = 1/(2*Lambda)*(1/diag_s_p)*(np.matmul(nu_h2, S_sq) - np.matmul(nu_h, S)**2)
      #print('nu', nu(self, a, Lambda))
      #print('First', np.matmul(nu_h2, S_sq))
      #print('Second', np.matmul(nu_h, S)**2)
      #print('S', S)
      L = First_term + second_term
      #print('L_compute:', L)
      return L#[0]

    def L(self, a, m, Sigma, Lambda):
      """ Calls on above functions to compute learning term L """
      (coeff_first_term, S_sq, S, g, h, diag_s_p) = L_input(self,m, Sigma, Lambda)
      L = L_compute(self,a, Lambda, coeff_first_term, S_sq, S, g, h, diag_s_p)
      #print('L:', L)
      return L

    def f(self,m, Sigma): #d) Have replaced argument d with S
      """
      Computes f, the vector of the conditional expected values of the
      rewards (conditional on the current values of m and Sigma)

      """
      #Sigma = self.Sigma(d)
      B = self.B
      P = self.P
      S = np.array(np.matmul(np.matmul(B, Sigma), np.transpose(B)))
      #print('S:', S)
      #print('B:',B)
      #print('m_t:', np.transpose(m))
      v_d = np.array(1 + np.diag(S) + 1/P)
      b_m = np.array(np.matmul(B,np.transpose(m)))
      #f = norm.cdf(b_m/np.sqrt(v_d))
      #f = f[0]
      f = b_m[0]            # Changed reward function to be linear
      return f

    def alpha(self, m, Sigma, f, Lambda, beta):
        """
        Numerically solves the fixed point problem required for the ARC algorithm.

        For details regarding the numerical methods used, see section 3.3 in the
        Research Summary.

        """

        def root_search(a):
            return f + (beta/(1-beta))*L(self,a,m,Sigma, Lambda) - a

        ts = time.time()
        a_root_obj = scipy.optimize.root(root_search, f + (beta/(1-beta))*L(self,f,m,Sigma, Lambda), method='df-sane', options={'maxfev':10, 'disp':True})#.x
        a_root = a_root_obj.x
        print(a_root_obj.message)
        print('Time for fixed point:', time.time()-ts)
        #a_root1=a_root
        if not a_root_obj.success:
            print('hybr method used')
            ts = time.time()
            a_root_obj = scipy.optimize.root(root_search, f + (beta/(1-beta))*L(self,f,m,Sigma, Lambda), method='hybr')
            a_root = a_root_obj.x
            print(a_root_obj.message)
            print(a_root_obj.success)
            print('Time for fixed point:', time.time()-ts)
            #print('Difference', a_root1-a_root)
        if not a_root_obj.success:
            a_root = f + (beta/(1-beta))*L(self,self.P,m,Sigma, Lambda)
        #except:
            #print('error')
        a_root_test = scipy.optimize.root(root_search, f + (beta/(1-beta))*L(self,self.P,m,Sigma, Lambda), method='hybr')
        print('test result:', a_root_test.message)
        print('test result:', a_root_test.success)
        print('Solution difference', a_root - a_root_test.x)

        a_root_test2 = scipy.optimize.root(root_search, f + (beta/(1-beta))*L(self,np.ones(self._num_actions),m,Sigma, Lambda), method='hybr')
        print('test2 result:', a_root_test2.message)
        print('test2 result:', a_root_test2.success)
        print('Solution difference', a_root - a_root_test2.x)
        return a_root

    def nu(self, a, Lambda):
      """
      The function nu in the ARC paper (Reference 1), which is used to
      calculate the probability distribution from which the action is chosen

      """
      a = np.array(a)
      #print(a)
      ref = max(a)
      W = np.exp((a-ref)/Lambda)          # subtract off max to make all a terms negative (doesn't matter though as it cancels - just for computation)
      K = sum(W)
      return  W/K
  #####################################################

    f = f(self,m,Sigma)                # This is an estimate of the rewards
    Lambda = rho*tf.norm(Sigma)
    Realised_reward = alpha(self, m, Sigma, f, Lambda, beta)     # solves for alpha (ie solves the fixed point problem) - this is a vector. (have replaced 1/n with S)
    Prob_simplex = nu(self, Realised_reward, Lambda)             # generates probability distribution (which is nu = S'(alpha/lambda) in the paper)

    print('B:', self.B)
    print('m:', m)
    print('f', f)
    print('alpha:', Realised_reward)
    print('L:', ((1-beta)/beta)*(Realised_reward-f))

    print('outcome', f + (beta/(1-beta))*L(self,self.P,m,Sigma, Lambda))

    print('Probabilities:', Prob_simplex)
    random_choice = np.random.choice(range(len(Prob_simplex)), p=Prob_simplex) # samples from the probability distribution
    print('Chosen Action:', random_choice)
    return random_choice

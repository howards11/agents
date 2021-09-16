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
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

tfd = tfp.distributions


def ARC_policy(self, m, Sigma, rho, beta, Verbose):
    """
    Function that computes the chosen action according to the ARC algorithm

    References:
    1.  "Asymptotic Randomised Control with applications to bandits",
        S. N. Cohen and T. Treetanthiploet, arXiv:2010.07252, 2020.

    2.  https://github.com/ttreetanthiploet/ARC-with-bandits

    """

    def L_input(self, m, Sigma, Lambda):
      """ Computes input variables for L """
      B = self.B
      P = self.P

      S = np.array(np.matmul(np.matmul(B, Sigma), np.transpose(B)))
      S_sq = S**2
      diag_s_p = np.array(np.diag(S) + 1/P)
      return (S_sq, S, diag_s_p)

    def f(self,m):
      """
      Computes f, the vector of the conditional expected values of the
      rewards (conditional on the current values of m and Sigma)

      """
      B = self.B
      b_m = np.array(np.matmul(B,np.transpose(m)))
      f = b_m[0]            # Changed reward function to be linear
      return f

    def alpha(self, m, Sigma, f, Lambda, beta):
        """
        Numerically solves the fixed point problem required for the ARC algorithm.

        For details regarding the numerical methods used, see section 3.3 in the
        Research Summary.

        """
        (S_sq, S, diag_s_p) = L_input(self, m, Sigma, Lambda)
        coeff = (beta/(1-beta))*1/(2*Lambda)*(1/diag_s_p)

        def root_search(a):
            return f + coeff*( np.matmul(nu(self, a, Lambda), S_sq) - np.matmul(nu(self, a, Lambda), S)**2 ) - a

        a_root_obj = scipy.optimize.root(root_search, f + coeff*( np.matmul(nu(self, f, Lambda), S_sq) - np.matmul(nu(self, f, Lambda), S)**2 ), method='df-sane', options={'maxfev':10, 'disp':self._Verbose})
        a_root = a_root_obj.x
        if self._Verbose:
            print(a_root_obj.message)

        if not a_root_obj.success:              # If df-sane fails, use hybrid
            a_root_obj = scipy.optimize.root(root_search, f + coeff*( np.matmul(nu(self, f, Lambda), S_sq) - np.matmul(nu(self, f, Lambda), S)**2 ), method='hybr')
            a_root = a_root_obj.x
            if self._Verbose:
                print('hybr method used')
                print('Second attempt:', a_root_obj.message)
                print('Second attempt success:', a_root_obj.success)


        if not a_root_obj.success:              # If above fails, start iterations using P
            a_root_obj = scipy.optimize.root(root_search, f + coeff*( np.matmul(nu(self, self.P, Lambda), S_sq) - np.matmul(nu(self, self.P, Lambda), S)**2 ), method='hybr')
            a_root = a_root_obj.x
            if self._Verbose:
                print('Third attempt:', a_root_obj.message)
                print('Third attempt success:', a_root_obj.success)

        if not a_root_obj.success:              # If above fails, start iterations using ones vector
            a_root_obj = scipy.optimize.root(root_search, f + coeff*( np.matmul(nu(self, np.ones(self._num_actions), Lambda), S_sq) - np.matmul(nu(self, np.ones(self._num_actions), Lambda), S)**2 ), method='hybr')
            a_root = a_root_obj.x
            if self._Verbose:
                print('Fourth attempt:', a_root_obj.message)
                print('Fourth attempt success:', a_root_obj.success)

        return a_root, a_root_obj.success

    def nu(self, a, Lambda):
      """
      The function nu in the ARC paper (Reference 1), which is used to
      calculate the probability distribution from which the action is chosen

      """
      a = np.array(a)
      ref = max(a)
      W = np.exp((a-ref)/Lambda)          # subtract off max to make all a terms negative (doesn't matter though as it cancels - just for computation)
      K = sum(W)
      return  W/K

    ############################################################################

    f = f(self,m)                # This is an estimate of the rewards
    Lambda = rho*tf.norm(Sigma)

    Realised_reward, success = alpha(self, m, Sigma, f, Lambda, beta)     # solves for alpha (ie solves the fixed point problem).
    Prob_simplex = nu(self, Realised_reward, Lambda)             # generates probability distribution (which is nu = S'(alpha/Lambda) in the paper)

    random_choice = np.random.choice(range(len(Prob_simplex)), p=Prob_simplex) # samples from the probability distribution

    if self._Verbose:
        print('Observation matrix B:', self.B)
        print('m:', m)
        print('f:', f)
        print('alpha:', Realised_reward)
        print('L:', ((1-beta)/beta)*(Realised_reward-f))
        print('Probabilities:', Prob_simplex)
        print('Chosen Action:', random_choice, self.B[random_choice])

    return random_choice, success

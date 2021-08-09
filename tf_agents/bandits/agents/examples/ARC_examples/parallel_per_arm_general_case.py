# coding=utf-8
# Copyright 2020 The TF-Agents Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" Example demonstrating the StationaryStochasticPerArmPyEnvironment case

*********************** ADDED TO ASSESS ARC PERFORMANCE ************************

This is the example in Figure ?? in the Research Summary

Add in your chosen file directories (file_dir_1, file_dir_2) to save the plots

"""

import numpy as np

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.agents import tf_agent
from tf_agents.drivers import driver
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.policies import tf_policy
from tf_agents.specs import array_spec
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tf_agents.trajectories import policy_step

# Imports for example.
from tf_agents.bandits.agents import lin_ucb_agent
from tf_agents.bandits.agents import linear_thompson_sampling_agent
from tf_agents.bandits.agents import lin_arc_agent

from tf_agents.bandits.environments import stationary_stochastic_per_arm_py_environment as p_a_env
from tf_agents.bandits.metrics import tf_metrics as tf_bandit_metrics
from tf_agents.drivers import dynamic_step_driver
from tf_agents.replay_buffers import tf_uniform_replay_buffer


import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import pickle


######################### DEFINE ENVIRONMENT PARAMETERS ########################

# The dimension of the global features (dimension l in the Research Summary).
GLOBAL_DIM = 10  #@param {type:"integer"}
# The elements of the global feature will be integers in [-GLOBAL_BOUND, GLOBAL_BOUND).
GLOBAL_BOUND = 10  #@param {type:"integer"}
# The dimension of the per-arm features (dimension k in the Research Summary).
PER_ARM_DIM = 20  #@param {type:"integer"}
# The elements of the PER-ARM feature will be integers in [-PER_ARM_BOUND, PER_ARM_BOUND).
PER_ARM_BOUND = 10  #@param {type:"integer"}
# The variance of the Gaussian distribution that generates the rewards.
VARIANCE = 20.0  #@param {type: "number"}
# The elements of the linear reward parameter will be integers in [-PARAM_BOUND, PARAM_BOUND).
PARAM_BOUND = 10  #@param {type: "integer"}

NUM_ACTIONS = 20  #@param {type:"integer"}      # Number of actions
BATCH_SIZE = 1  #@param {type:"integer"}        # Number of batches at each training step


ncpu = 1                                # For running simulations in parallel

Rep = 50                                 # Number of simulations
num_iterations = 100 # @param          # This is the HORIZON
steps_per_loop = 1 # @param

# Define which of the agents to use (possible agents are defined later)
agents = ['ucb', 'ts', 'arc_0_01']#, 'arc_0_1', 'arc_1', 'arc_10']


def global_context_sampling_fn():
  """
  This function generates a single global observation vector from
  [-GLOBAL_BOUND, GLOBAL_BOUND)^GlOBAL_DIM at each time step.
  """
  return np.random.randint(
      -GLOBAL_BOUND, GLOBAL_BOUND, [GLOBAL_DIM]).astype(np.float32)

def per_arm_context_sampling_fn():
  """"
  This function generates a single per-arm observation vector from
  [-PER_ARM_BOUND, PER_ARM_BOUND)^PER_ARM_DIM at each time step.
  """
  return np.random.randint(
      -PER_ARM_BOUND, PER_ARM_BOUND, [PER_ARM_DIM]).astype(np.float32)


############################# DEFINE RUN FUNCTION ##############################

def run(rep):
    print('Repetition', rep)

    ########## DEFINE HIDDEN PARAMETERS, REWARD FNS, ENVIRONMENT ##########

    reward_param = list(np.random.randint(
          -PARAM_BOUND, PARAM_BOUND, [GLOBAL_DIM + PER_ARM_DIM]))

    def linear_normal_reward_fn(x):
      """This function generates a reward from the concatenated global and per-arm observations."""
      mu = np.dot(x, reward_param)
      return np.random.normal(mu, VARIANCE)

    per_arm_py_env = p_a_env.StationaryStochasticPerArmPyEnvironment(
        global_context_sampling_fn,
        per_arm_context_sampling_fn,
        NUM_ACTIONS,
        linear_normal_reward_fn,
        batch_size=BATCH_SIZE
    )
    per_arm_tf_env = tf_py_environment.TFPyEnvironment(per_arm_py_env)

    ####################### DEFINE SPECS #######################

    observation_spec = per_arm_tf_env.observation_spec()
    time_step_spec = ts.time_step_spec(observation_spec)
    action_spec = tensor_spec.BoundedTensorSpec(
        dtype=tf.int32, shape=(), minimum=0, maximum=NUM_ACTIONS - 1)

    ################## DEFINE POSSIBLE AGENTS ##################

    ucb_agent = lin_ucb_agent.LinearUCBAgent(time_step_spec=time_step_spec,
                                         action_spec=action_spec,
                                         accepts_per_arm_features=True,
                                         name='ucb')

    ts_agent = linear_thompson_sampling_agent.LinearThompsonSamplingAgent(time_step_spec=time_step_spec,
                                         action_spec=action_spec,
                                         accepts_per_arm_features=True,
                                         name = 'ts')

    arc_agent_0_01 = lin_arc_agent.LinearARCAgent(time_step_spec=time_step_spec,
                                         action_spec=action_spec,
                                         accepts_per_arm_features=True,
                                         name='arc_0_01',
                                         rho = 0.01,
                                         beta = 1-1/num_iterations)

    arc_agent_0_1 = lin_arc_agent.LinearARCAgent(time_step_spec=time_step_spec,
                                         action_spec=action_spec,
                                         accepts_per_arm_features=True,
                                         name='arc_0_1',
                                         rho = 0.1,
                                         beta = 1-1/num_iterations)

    arc_agent_1 = lin_arc_agent.LinearARCAgent(time_step_spec=time_step_spec,
                                         action_spec=action_spec,
                                         accepts_per_arm_features=True,
                                         name='arc_1',
                                         rho = 1,
                                         beta = 1-1/num_iterations)
    arc_agent_10 = lin_arc_agent.LinearARCAgent(time_step_spec=time_step_spec,
                                         action_spec=action_spec,
                                         accepts_per_arm_features=True,
                                         name='arc_10',
                                         rho = 10,
                                         beta = 1-1/num_iterations)

    ##################### DEFINE METRICS #####################

    def _all_rewards(observation, hidden_param):
      """Outputs rewards for all actions, given an observation."""
      hidden_param = tf.cast(hidden_param, dtype=tf.float32)
      global_obs = observation['global']
      per_arm_obs = observation['per_arm']
      num_actions = tf.shape(per_arm_obs)[1]
      tiled_global = tf.tile(
          tf.expand_dims(global_obs, axis=1), [1, num_actions, 1])
      concatenated = tf.concat([tiled_global, per_arm_obs], axis=-1)
      rewards = tf.linalg.matvec(concatenated, hidden_param)
      print('rewards:', rewards)
      return rewards

    def optimal_reward(observation):
      """Outputs the maximum expected reward for every element in the batch."""
      return tf.reduce_max(_all_rewards(observation, reward_param), axis=1)

    def optimal_action(observation):
      """Outputs the action with the best expected reward for every element in the batch."""
      return tf.argmax(_all_rewards(observation, reward_param), axis=1, output_type=tf.int32)

    def find_chosen_reward(observation, action):
      """Outputs the expected reward for the chosen element in the batch."""
      #return _all_rewards(observation, reward_param)[action]
      global_obs = observation['global']
      per_arm_obs = observation['per_arm']
      num_actions = tf.shape(per_arm_obs)[1]
      tiled_global = tf.tile(
          tf.expand_dims(global_obs, axis=1), [1, num_actions, 1])
      concatenated = tf.concat([tiled_global, per_arm_obs], axis=-1)

      chosen_obs = tf.gather(concatenated, action, batch_dims=1)
      return tf.linalg.matvec(chosen_obs, reward_param)

    #regret_metric = tf_bandit_metrics.RegretMetric(optimal_reward)
    regret_metric = tf_bandit_metrics.ExpectedRegretMetric(optimal_reward, find_chosen_reward)
    SubOpt_metric = tf_bandit_metrics.SuboptimalArmsMetric(optimal_action)
    #regret_metric = tf_bandit_metrics.ExpectedRegretMetric(optimal_reward, find_chosen_reward)


    ############ RUN THE SIMULATION FOR EACH OF THE AGENTS ############

    possible_agents = [ucb_agent, ts_agent, arc_agent_0_01, arc_agent_0_1, arc_agent_1, arc_agent_10]

    used_agents = []
    for agent_name in agents:
        try:
            for agent in possible_agents:
                if agent.name == agent_name:
                    used_agents.append(agent)
                    break
        except:
            print('Error: no such agent', agent_name, 'defined')


    regret_lst = []
    SubOpt_lst = []

    for i in range(len(agents)):

        agent = used_agents[i]

        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(             # Stores experience
            data_spec=agent.policy.trajectory_spec,
            batch_size=BATCH_SIZE,
            max_length=steps_per_loop)

        observers = [replay_buffer.add_batch, regret_metric, SubOpt_metric]

        driver = dynamic_step_driver.DynamicStepDriver(
            env=per_arm_tf_env,
            policy=agent.collect_policy,
            num_steps=steps_per_loop * BATCH_SIZE,
            observers=observers)

        regret_values = []
        SubOpt_values = []

        for _ in range(num_iterations):
            #print('Iteration', _)
            driver.run()
            loss_info = agent.train(replay_buffer.gather_all())           # Trains agent using the experience in the buffer
            replay_buffer.clear()
            regret_values.append(regret_metric.result())
            SubOpt_values.append(SubOpt_metric.result())
            print('Repetition', rep)
            print('reward_param', reward_param)
            print('--------------')

        regret_lst.append(regret_values)
        SubOpt_lst.append(SubOpt_values)

    return (regret_lst, SubOpt_lst)

############### RUN THE SIMULATIONS IN PARALLEL, PICKLE RESULTS ################

H = Parallel(n_jobs=ncpu, max_nbytes='10M')(delayed(run)(j) for j in range(Rep))

#previous_H = pickle.load( open('per_arm_general_case_data', "rb" ) )
#try:
#    print('previous_H shape', previous_H.shape)
#except:
#    print('previous_H shape', len(previous_H))

#H = [*previous_H, *H]

#try:
#    print('new_H shape', H.shape)
#except:
#    print('new_H shape', len(H))

pickle.dump(H, open('per_arm_general_case_data', "wb" ) )

################################ READ RESULTS ##################################

Data = pickle.load( open('per_arm_general_case_data', "rb" ) )

Rep = len(Data)

Regret_record = np.zeros((len(agents), Rep, num_iterations))
SubOpt_record = np.zeros((len(agents), Rep, num_iterations))

for rep in range(Rep):
    Regret_record[:, rep, :] = Data[rep][0]
    SubOpt_record[:, rep, :] = Data[rep][1]


################################ PLOT RESULTS ##################################
fig, axs = plt.subplots(2, 2)

# Compute the cumulative regret for each agent
Cumul_regret = np.cumsum(Regret_record, axis = 2)

# Compute the results to plot
mean_cumul_regret = np.mean(Cumul_regret, axis = 1)
median_cumul_regret = np.median(Cumul_regret, axis = 1)
q75_cumul_regret = np.quantile(Cumul_regret, 0.75, axis = 1)
q90_cumul_regret = np.quantile(Cumul_regret, 0.90, axis = 1)

# Plot results
for i in range(len(agents)):
    axs[0,0].plot(np.arange(1,num_iterations+1), mean_cumul_regret[i,:], label = agents[i])
    axs[0,0].set_title('Mean of regret', fontsize = 8)
    axs[0,0].legend(loc = 'lower right', fontsize=6)

    axs[0,1].plot(np.arange(1,num_iterations+1), median_cumul_regret[i,:], label = agents[i])
    axs[0,1].set_title('Median of regret', fontsize = 8)
    axs[0,1].legend(loc = 'lower right', fontsize=6)

    axs[1,0].plot(np.arange(1,num_iterations+1), q75_cumul_regret[i,:], label = agents[i])
    axs[1,0].set_title('0.75 quantile of regret', fontsize = 8)
    axs[1,0].legend(loc = 'lower right', fontsize=6)

    axs[1,1].plot(np.arange(1,num_iterations+1), q90_cumul_regret[i,:], label = agents[i])
    axs[1,1].set_title('0.90 quantile of regret', fontsize = 8)
    axs[1,1].legend(loc = 'lower right', fontsize=6)

fig.tight_layout()
fig.suptitle('Regret metrics from '+str(Rep)+' simulations', fontsize = 10)
fig.subplots_adjust(top=0.88)
plt.savefig(r'file_dir_1')


fig, axs = plt.subplots(2, 2)

# Compute the cumulative SubOpt metric for each agent
Cumul_SubOpt = np.cumsum(SubOpt_record, axis = 2)

# Compute the results to plot
mean_cumul_SubOpt = np.mean(Cumul_SubOpt, axis = 1)
median_cumul_SubOpt = np.median(Cumul_SubOpt, axis = 1)
q75_cumul_SubOpt = np.quantile(Cumul_SubOpt, 0.75, axis = 1)
q90_cumul_SubOpt = np.quantile(Cumul_SubOpt, 0.90, axis = 1)

# Plot results
for i in range(len(agents)):
    axs[0,0].plot(np.arange(1,num_iterations+1), mean_cumul_SubOpt[i,:], label = agents[i])
    axs[0,0].set_title('Mean of cumulative SubOpt arms', fontsize = 8)
    axs[0,0].legend(loc = 'upper left', fontsize=6)

    axs[0,1].plot(np.arange(1,num_iterations+1), median_cumul_SubOpt[i,:], label = agents[i])
    axs[0,1].set_title('Median of cumulative SubOpt arms', fontsize = 8)
    axs[0,1].legend(loc = 'upper left', fontsize=6)

    axs[1,0].plot(np.arange(1,num_iterations+1), q75_cumul_SubOpt[i,:], label = agents[i])
    axs[1,0].set_title('0.75 quantile of cumul. SubOpt arms', fontsize = 8)
    axs[1,0].legend(loc = 'upper left', fontsize=6)

    axs[1,1].plot(np.arange(1,num_iterations+1), q90_cumul_SubOpt[i,:], label = agents[i])
    axs[1,1].set_title('0.90 quantile of cumul. SubOpt arms', fontsize = 8)
    axs[1,1].legend(loc = 'upper left', fontsize=6)

fig.tight_layout()
fig.suptitle('SubOpt Metrics from '+str(Rep)+' simulations', fontsize = 10)
fig.subplots_adjust(top=0.88)
plt.savefig(r'file_dir_2')

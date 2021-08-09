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

""" Example demonstrating the Graphical Bandit case

*********************** ADDED TO ASSESS ARC PERFORMANCE ************************

This is the example in Figure ?? in the Research Summary

Add in your chosen file directories (file_dir_1, file_dir_2, file_dir_3,
file_dir_4) to save the plots

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

import networkx as nx


######################### DEFINE ENVIRONMENT PARAMETERS ########################

NUM_ACTIONS = 10  #@param {type:"integer"}      # Number of actions
BATCH_SIZE = 1  #@param {type:"integer"}        # Number of batches at each training step

GRAPH_SIZE = 10         # Number of nodes on graph
P_VALUE = 0.4           # Value of p to use when creating a random graph


ncpu = 3                # For running simulations in parallel

Rep = 3                            # Number of simulations
num_iterations = 40 # @param      # This is the HORIZON
steps_per_loop = 1 # @param

# Define which of the agents to use (possible agents are defined later)
agents = ['ucb', 'ts', 'arc_0_1']#, 'arc_1']
#agents = ['arc_0_1']

# global_context_sampling_fn is not used for the graphical bandit
GlOBAL_DIM = 0
GLOBAL_BOUND = 10
def global_context_sampling_fn():
  """This function generates a single global observation vector."""
  return np.random.randint(
      -GLOBAL_BOUND, GLOBAL_BOUND, [GlOBAL_DIM]).astype(np.float32)

############################# DEFINE RUN FUNCTION ##############################

def run(rep):
    print('Repetition', rep)

    ### DEFINE GRAPH, HIDDEN PARAMETERS, REWARD FNS, VARIANCE FN, ENVIRONMENT ##

    # Create hidden parameter and variance parameter, one entry for each node of the graph
    reward_param = list(2*np.random.rand(GRAPH_SIZE)-np.ones(GRAPH_SIZE))
    variance_vector = list(np.ones(GRAPH_SIZE)/5)

    # Create random binomial graph using the given parameters
    G = nx.generators.binomial_graph(GRAPH_SIZE, P_VALUE)
    plt.cla()
    nx.draw(G, with_labels=True)
    plt.savefig(r'file_dir_1')


    def per_arm_context_sampling_fn():
        """"This function generates a single per-arm observation vector."""
        degrees = np.array([val for (node, val) in G.degree()])
        #node = np.random.randint(0, GRAPH_SIZE)       # Which vertex we are centred on
        node = np.random.choice(GRAPH_SIZE, p = degrees/degrees.sum())
        neighbours = [n for n in G[node]]               # list of neighbours of the centre node
        obs = np.zeros(GRAPH_SIZE)
        obs[node] = 1

        for n in neighbours:                            # choose adjacent vertices with probability 0.5
            seed = np.random.random()
            if seed < 0.4:
                obs[n] = 1

        return obs.astype(np.float32)

    def linear_normal_reward_fn(x):
      """This function generates a reward from the concatenated global and per-arm observations."""
      mu = np.dot(x, reward_param)
      #if np.sum(x)==2:
        #  var = 0.01
      #else:
        #  var = 1
      #var = np.sum(x)**4
      #def variance_fn(x):
          #return np.dot(x, variance_vector)**3
        #  if np.sum(x)==1:
        #      var = 0.001
         # if np.sum(x)==2:
        #      var = 0.01
         # if np.sum(x)==3:
        #      var = 0.5
         # if np.sum(x)>3:
        #      var = 1
         # return var

      #var = variance_fn(x)
      var = np.sum(x)/100
      return np.random.normal(mu, var)

    def variance_fn(x):
        """This function defines the variance in the reward of an arm with observation x"""
        #return np.dot(x, variance_vector)**3
        var = np.sum(x)/100
        return var

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
                                         variance_fn = variance_fn,
                                         name='ucb')

    ts_agent = linear_thompson_sampling_agent.LinearThompsonSamplingAgent(time_step_spec=time_step_spec,
                                         action_spec=action_spec,
                                         accepts_per_arm_features=True,
                                         variance_fn = variance_fn,
                                         name = 'ts')

    arc_agent_0_01 = lin_arc_agent.LinearARCAgent(time_step_spec=time_step_spec,
                                         action_spec=action_spec,
                                         accepts_per_arm_features=True,
                                         variance_fn = variance_fn,
                                         name='arc_0_01',
                                         rho = 0.01,
                                         beta = 1-1/num_iterations)

    arc_agent_0_1 = lin_arc_agent.LinearARCAgent(time_step_spec=time_step_spec,
                                         action_spec=action_spec,
                                         accepts_per_arm_features=True,

                                         variance_fn = variance_fn,

                                         name='arc_0_1',
                                         rho = 0.1,
                                         beta = 1-1/num_iterations)

    arc_agent_1 = lin_arc_agent.LinearARCAgent(time_step_spec=time_step_spec,
                                         action_spec=action_spec,
                                         accepts_per_arm_features=True,
                                         variance_fn = variance_fn,
                                         name='arc_1',
                                         rho = 0.1,
                                         beta = 0.995)#1-1/num_iterations)

    arc_agent_10 = lin_arc_agent.LinearARCAgent(time_step_spec=time_step_spec,
                                         action_spec=action_spec,
                                         accepts_per_arm_features=True,
                                         variance_fn = variance_fn,
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
      action = tf.argmax(_all_rewards(observation, reward_param), axis=1, output_type=tf.int32)

      global_obs = observation['global']
      per_arm_obs = observation['per_arm']
      num_actions = tf.shape(per_arm_obs)[1]
      tiled_global = tf.tile(
          tf.expand_dims(global_obs, axis=1), [1, num_actions, 1])
      concatenated = tf.concat([tiled_global, per_arm_obs], axis=-1)

      #arms = [i for i in range(len(reward_param)) if tf.equal(tf.gather(concatenated, i), tf.gather(concatenated, action))]
      return tf.gather(concatenated, action, batch_dims=1)

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
    SubOpt_metric = tf_bandit_metrics.SuboptimalArmsMetricNew(optimal_action)
    obs_sum_tracker = tf_bandit_metrics.ObsSum()

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
    obs_sum_lst = []

    for i in range(len(agents)):

        agent = used_agents[i]

        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(             # Stores experience
            data_spec=agent.policy.trajectory_spec,
            batch_size=BATCH_SIZE,
            max_length=steps_per_loop)

        observers = [replay_buffer.add_batch, regret_metric, SubOpt_metric, obs_sum_tracker]

        driver = dynamic_step_driver.DynamicStepDriver(
            env=per_arm_tf_env,
            policy=agent.collect_policy,
            num_steps=steps_per_loop * BATCH_SIZE,
            observers=observers)

        regret_values = []
        SubOpt_values = []
        obs_sum_values = []

        for _ in range(num_iterations):
            #print('Iteration', _)
            driver.run()
            loss_info = agent.train(replay_buffer.gather_all())           # Trains agent using the experience in the buffer
            #print(replay_buffer.gather_all())
            replay_buffer.clear()
            regret_values.append(regret_metric.result())
            SubOpt_values.append(SubOpt_metric.result())
            obs_sum_values.append(obs_sum_tracker.result())
            print('Repetition', rep)
            print('reward_param', reward_param)
            print('--------------')

        regret_lst.append(regret_values)
        SubOpt_lst.append(SubOpt_values)
        obs_sum_lst.append(obs_sum_values)

    return (regret_lst, SubOpt_lst, obs_sum_lst)

############### RUN THE SIMULATIONS IN PARALLEL, PICKLE RESULTS ################

H = Parallel(n_jobs=ncpu, max_nbytes='10M')(delayed(run)(j) for j in range(Rep))

#previous_H = pickle.load( open('graphical_bandit_2_data', "rb" ) )
#try:
#    print('previous_H shape', previous_H.shape)
#except:
#    print('previous_H shape', len(previous_H))
#
#H = [*previous_H, *H]

#try:
#    print('new_H shape', H.shape)
#except:
#    print('new_H shape', len(H))

pickle.dump(H, open('graphical_bandit_2_data', "wb" ) )

################################ READ RESULTS ##################################

Data = pickle.load( open('graphical_bandit_2_data', "rb" ) )

Rep = len(Data)

Regret_record = np.zeros((len(agents), Rep, num_iterations))
SubOpt_record = np.zeros((len(agents), Rep, num_iterations))
obs_sum_record = np.zeros((len(agents), Rep, num_iterations))

for rep in range(Rep):
    Regret_record[:, rep, :] = Data[rep][0]
    SubOpt_record[:, rep, :] = Data[rep][1]
    obs_sum_record[:, rep, :] = Data[rep][2]


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
    axs[0,0].legend(loc = 'upper left', fontsize=6)

    axs[0,1].plot(np.arange(1,num_iterations+1), median_cumul_regret[i,:], label = agents[i])
    axs[0,1].set_title('Median of regret', fontsize = 8)
    axs[0,1].legend(loc = 'upper left', fontsize=6)

    axs[1,0].plot(np.arange(1,num_iterations+1), q75_cumul_regret[i,:], label = agents[i])
    axs[1,0].set_title('0.75 quantile of regret', fontsize = 8)
    axs[1,0].legend(loc = 'upper left', fontsize=6)

    axs[1,1].plot(np.arange(1,num_iterations+1), q90_cumul_regret[i,:], label = agents[i])
    axs[1,1].set_title('0.90 quantile of regret', fontsize = 8)
    axs[1,1].legend(loc = 'upper left', fontsize=6)

fig.tight_layout()
fig.suptitle('Regret metrics from '+str(Rep)+' simulations', fontsize = 10)
fig.subplots_adjust(top=0.88)
plt.savefig(r'file_dir_2')



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
plt.savefig(r'file_dir_3')



fig, axs = plt.subplots(1)
mean_obs_sum = np.mean(obs_sum_record, axis = 1)
plt.cla()
for i in range(len(agents)):
    plt.plot(np.arange(1,30+1), mean_obs_sum[i,:30], label = agents[i])
plt.ylabel('Mean Sum of Chosen Observations')
plt.xlabel('Trials t')
plt.legend(loc = 'lower right')
plt.savefig(r'file_dir_4')

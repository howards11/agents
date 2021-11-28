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

""" Example demonstrating the StationaryStochasticPyEnvironment case

*********************** ADDED TO ASSESS ARC PERFORMANCE ************************

This is the example in Figures 4 and 5 in the Research Summary

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

from tf_agents.bandits.environments import stationary_stochastic_py_environment as sspe
from tf_agents.bandits.metrics import tf_metrics as tf_bandit_metrics
from tf_agents.drivers import dynamic_step_driver
from tf_agents.replay_buffers import tf_uniform_replay_buffer

import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import pickle


######################### DEFINE ENVIRONMENT PARAMETERS ########################

batch_size = 1 # @param         # Number of batches at each training step

No_arms = 6                     # Number of actions
param_dimension = 5             # Dimension l in the Research Summary
PARAM_BOUND = 10                # Entries of context and hidden parameters are sampled in [-PARAM_BOUND, PARAM_BOUND]
VARIANCE = 10                   # Variance in the rewards

ncpu = 5                        # For running simulations in parallel

Rep = 1000                      # Number of simulations
num_iterations = 200 # @param   # This is the HORIZON (value of T)
steps_per_loop = 1 # @param

use_previous_sims = False       # Whether to use previously ran simulation results

# Define which of the agents to use (possible agents are defined later)
agents = ['ucb_0_1', 'ucb_0_5', 'ucb_1', 'ucb_1_5',
        'ts_0_1', 'ts_0_5', 'ts_1', 'ts_1_5',
        'arc_0_01', 'arc_0_1', 'arc_1', 'arc_10']

def context_sampling_fn(batch_size):
  """Samples the context from [-PARAM_BOUND, PARAM_BOUND)^param_dimension at each time step."""
  def _context_sampling_fn():
    return np.random.randint(-PARAM_BOUND, PARAM_BOUND, [batch_size, param_dimension]).astype(np.float32)        # The way the observations (context) are generated comes from this
  return _context_sampling_fn

class LinearNormalReward(object):
  """A class that acts as linear reward function when called."""
  def __init__(self, theta, sigma):
    self.theta = theta
    self.sigma = sigma
  def __call__(self, x):
    mu = np.dot(x, self.theta)                   # Mean mu is dot product of x and theta
    return np.random.normal(mu, self.sigma)


################################# DEFINE SPECS #################################

observation_spec = tensor_spec.TensorSpec([param_dimension], tf.float32)  # Gives dimensions, type of the observations
time_step_spec = ts.time_step_spec(observation_spec)
action_spec = tensor_spec.BoundedTensorSpec(
    dtype=tf.int32, shape=(), minimum=0, maximum=No_arms-1)         # Gives info about the possible actions


############################# DEFINE RUN FUNCTION ##############################

def run(rep):
    print('Repetition', rep)

    ########## DEFINE HIDDEN PARAMETERS, REWARD FNS, ENVIRONMENT ##########

    arm_params = []
    arm_reward_fns = []

    for i in range(No_arms):
        param = list(np.random.randint(
              -PARAM_BOUND, PARAM_BOUND, [param_dimension]))
        arm_params.append(param)
        arm_reward_fns.append(LinearNormalReward(param, VARIANCE))

    print('Hidden parameters for arms:', arm_params)

    environment = tf_py_environment.TFPyEnvironment(
        sspe.StationaryStochasticPyEnvironment(
            context_sampling_fn(batch_size),
            arm_reward_fns,
            batch_size=batch_size))

    ############## DEFINE POSSIBLE AGENTS ##################

    ucb_agent_0_1 = lin_ucb_agent.LinearUCBAgent(time_step_spec=time_step_spec,
                                         action_spec=action_spec,
                                         name='ucb_0_1',
                                         alpha = 0.1)


    ucb_agent_0_5 = lin_ucb_agent.LinearUCBAgent(time_step_spec=time_step_spec,
                                         action_spec=action_spec,
                                         name='ucb_0_5',
                                         alpha = 0.5)

    ucb_agent_1 = lin_ucb_agent.LinearUCBAgent(time_step_spec=time_step_spec,
                                         action_spec=action_spec,
                                         name='ucb_1',
                                         alpha = 1)

    ucb_agent_1_5 = lin_ucb_agent.LinearUCBAgent(time_step_spec=time_step_spec,
                                         action_spec=action_spec,
                                         name='ucb_1_5',
                                         alpha = 1.5)


    ucb_agent_2 = lin_ucb_agent.LinearUCBAgent(time_step_spec=time_step_spec,
                                         action_spec=action_spec,
                                         name='ucb_2',
                                         alpha = 2)

    ts_agent_0_1 = linear_thompson_sampling_agent.LinearThompsonSamplingAgent(time_step_spec=time_step_spec,
                                         action_spec=action_spec,
                                         name = 'ts_0_1',
                                         alpha = 0.1)

    ts_agent_0_5 = linear_thompson_sampling_agent.LinearThompsonSamplingAgent(time_step_spec=time_step_spec,
                                         action_spec=action_spec,
                                         name = 'ts_0_5',
                                         alpha = 0.5)

    ts_agent_1 = linear_thompson_sampling_agent.LinearThompsonSamplingAgent(time_step_spec=time_step_spec,
                                         action_spec=action_spec,
                                         name = 'ts_1',
                                         alpha = 1)

    ts_agent_1_5 = linear_thompson_sampling_agent.LinearThompsonSamplingAgent(time_step_spec=time_step_spec,
                                         action_spec=action_spec,
                                         name = 'ts_1_5',
                                         alpha = 1.5)

    ts_agent_2 = linear_thompson_sampling_agent.LinearThompsonSamplingAgent(time_step_spec=time_step_spec,
                                         action_spec=action_spec,
                                         name = 'ts_2',
                                         alpha = 2)

    arc_agent_0_001 = lin_arc_agent.LinearARCAgent(time_step_spec=time_step_spec,
                                         action_spec=action_spec,
                                         name='arc_0_001',
                                         rho = 0.001,
                                         beta = 1-1/num_iterations)

    arc_agent_0_01 = lin_arc_agent.LinearARCAgent(time_step_spec=time_step_spec,
                                         action_spec=action_spec,
                                         name='arc_0_01',
                                         rho = 0.01,
                                         beta = 1-1/num_iterations)

    arc_agent_0_1 = lin_arc_agent.LinearARCAgent(time_step_spec=time_step_spec,
                                         action_spec=action_spec,
                                         name='arc_0_1',
                                         rho = 0.1,
                                         beta = 1-1/num_iterations)

    arc_agent_1 = lin_arc_agent.LinearARCAgent(time_step_spec=time_step_spec,
                                         action_spec=action_spec,
                                         name='arc_1',
                                         rho = 1,
                                         beta = 1-1/num_iterations)

    arc_agent_10 = lin_arc_agent.LinearARCAgent(time_step_spec=time_step_spec,
                                         action_spec=action_spec,
                                         name='arc_10',
                                         rho = 10,
                                         beta = 1-1/num_iterations)

    ##################### DEFINE METRICS #####################

    def compute_optimal_reward(observation):
      expected_reward_for_arms = []
      for i in range(No_arms):
          expected_reward_for_arms.append(tf.linalg.matvec(observation, tf.cast(arm_params[i], dtype=tf.float32)))
      optimal_action_reward = tf.reduce_max(expected_reward_for_arms, axis=0)
      return optimal_action_reward

    def compute_optimal_action(observation):
      expected_reward_for_arms = []
      for i in range(No_arms):
          expected_reward_for_arms.append(tf.linalg.matvec(observation, tf.cast(arm_params[i], dtype=tf.float32)))
      optimal_action = tf.argmax(expected_reward_for_arms, axis=0, output_type=tf.int32)
      return optimal_action

    def find_chosen_reward(observation, action):
      """Outputs the expected reward for the chosen element in the batch."""
      arm_params_tensor = tf.convert_to_tensor(arm_params)
      chosen_reward = tf.linalg.matvec(observation, tf.cast(tf.gather(arm_params_tensor, action, batch_dims=0), dtype=tf.float32))
      return chosen_reward

    #regret_metric = tf_metrics.RegretMetric(compute_optimal_reward)
    regret_metric = tf_bandit_metrics.ExpectedRegretMetric(compute_optimal_reward, find_chosen_reward)
    SubOpt_metric = tf_bandit_metrics.SuboptimalArmsMetric(compute_optimal_action)


    ############ RUN THE SIMULATION FOR EACH OF THE AGENTS ############

    possible_agents = [ucb_agent_0_1, ucb_agent_0_5, ucb_agent_1, ucb_agent_1_5, ucb_agent_2,
                    ts_agent_0_1, ts_agent_0_5, ts_agent_1, ts_agent_1_5, ts_agent_2,
                    arc_agent_0_001, arc_agent_0_01, arc_agent_0_1, arc_agent_1, arc_agent_10]

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

        print('Running simulation', rep, 'with agent', agent.name)

        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(             # Stores experience
            data_spec=agent.policy.trajectory_spec,
            batch_size=batch_size,
            max_length=steps_per_loop)

        observers = [replay_buffer.add_batch, regret_metric, SubOpt_metric]

        driver = dynamic_step_driver.DynamicStepDriver(
            env=environment,
            policy=agent.collect_policy,
            num_steps=steps_per_loop * batch_size,
            observers=observers)

        regret_values = []
        SubOpt_values = []

        for _ in range(num_iterations):
            driver.run()
            loss_info = agent.train(replay_buffer.gather_all())           # Trains agent using the experience in the buffer
            replay_buffer.clear()
            regret_values.append(regret_metric.result())
            SubOpt_values.append(SubOpt_metric.result())

        regret_lst.append(regret_values)
        SubOpt_lst.append(SubOpt_values)

    return (regret_lst, SubOpt_lst)


############### RUN THE SIMULATIONS IN PARALLEL, PICKLE RESULTS ################

if use_previous_sims:
    previous_H = pickle.load( open('sspe_general_case_data', "rb" ) )
    n_previous_sims = len(previous_H)

    if Rep >= n_previous_sims:
        print('Number of previous simulations:', n_previous_sims)
        H = Parallel(n_jobs=ncpu, max_nbytes='10M')(delayed(run)(j) for j in range(Rep-n_previous_sims))
        H = [*previous_H, *H]
        print('Total number of simulations:', len(H))
    else:
        print('Error: More existing simulations than value of Rep')
        H = previous_H
else:
    H = Parallel(n_jobs=ncpu, max_nbytes='10M')(delayed(run)(j) for j in range(Rep))

pickle.dump(H, open('sspe_general_case_data', "wb" ) )

################################ READ RESULTS ##################################

Data = pickle.load( open('sspe_general_case_data', "rb" ) )

Regret_record = np.zeros((len(agents), Rep, num_iterations))
SubOpt_record = np.zeros((len(agents), Rep, num_iterations))

for rep in range(Rep):
    Regret_record[:, rep, :] = Data[rep][0]
    SubOpt_record[:, rep, :] = Data[rep][1]


################################ PLOT RESULTS ##################################
fig, axs = plt.subplots(2, 2, figsize=(8,5))

colors = ['c', 'c', 'c', 'c', 'C1', 'C1', 'C1', 'C1', 'C2', 'C2', 'C2', 'C2', ]
linestyles = ['-', '--', ':', '-.', '-', '--', ':', '-.', '-', '--', ':', '-.']

# Compute the cumulative regret for each agent
Cumul_regret = np.cumsum(Regret_record, axis = 2)

# Compute the results to plot
mean_cumul_regret = np.mean(Cumul_regret, axis = 1)
median_cumul_regret = np.median(Cumul_regret, axis = 1)
q75_cumul_regret = np.quantile(Cumul_regret, 0.75, axis = 1)
q90_cumul_regret = np.quantile(Cumul_regret, 0.90, axis = 1)

# Plot results
for i in range(len(agents)):
    axs[0,0].plot(np.arange(1,num_iterations+1), mean_cumul_regret[i,:], label = agents[i], color = colors[i], linestyle = linestyles[i])
    axs[0,0].set_title('Mean of regret', fontsize = 8)

    axs[0,1].plot(np.arange(1,num_iterations+1), median_cumul_regret[i,:], label = agents[i], color = colors[i], linestyle = linestyles[i])
    axs[0,1].set_title('Median of regret', fontsize = 8)

    axs[1,0].plot(np.arange(1,num_iterations+1), q75_cumul_regret[i,:], label = agents[i], color = colors[i], linestyle = linestyles[i])
    axs[1,0].set_title('0.75 quantile of regret', fontsize = 8)

    axs[1,1].plot(np.arange(1,num_iterations+1), q90_cumul_regret[i,:], label = agents[i], color = colors[i], linestyle = linestyles[i])
    axs[1,1].set_title('0.90 quantile of regret', fontsize = 8)

fig.legend(loc='center left', labels=agents)

fig.tight_layout()
fig.suptitle('Regret metrics from '+str(Rep)+' simulations', fontsize = 12)
fig.subplots_adjust(top=0.88, left=0.25)
plt.savefig(r'file_dir_1')


fig, axs = plt.subplots(2, 2, figsize=(8,5))

# Compute the cumulative SubOpt metric for each agent
Cumul_SubOpt = np.cumsum(SubOpt_record, axis = 2)

# Compute the results to plot
mean_cumul_SubOpt = np.mean(Cumul_SubOpt, axis = 1)
median_cumul_SubOpt = np.median(Cumul_SubOpt, axis = 1)
q75_cumul_SubOpt = np.quantile(Cumul_SubOpt, 0.75, axis = 1)
q90_cumul_SubOpt = np.quantile(Cumul_SubOpt, 0.90, axis = 1)

# Plot results
for i in range(len(agents)):
    axs[0,0].plot(np.arange(1,num_iterations+1), mean_cumul_SubOpt[i,:], label = agents[i], color = colors[i], linestyle = linestyles[i])
    axs[0,0].set_title('Mean of cumulative SubOpt arms', fontsize = 8)
    axs[0,0].legend(loc = 'lower right', fontsize=6)

    axs[0,1].plot(np.arange(1,num_iterations+1), median_cumul_SubOpt[i,:], label = agents[i], color = colors[i], linestyle = linestyles[i])
    axs[0,1].set_title('Median of cumulative SubOpt arms', fontsize = 8)
    axs[0,1].legend(loc = 'lower right', fontsize=6)

    axs[1,0].plot(np.arange(1,num_iterations+1), q75_cumul_SubOpt[i,:], label = agents[i], color = colors[i], linestyle = linestyles[i])
    axs[1,0].set_title('0.75 quantile of cumul. SubOpt arms', fontsize = 8)
    axs[1,0].legend(loc = 'lower right', fontsize=6)

    axs[1,1].plot(np.arange(1,num_iterations+1), q90_cumul_SubOpt[i,:], label = agents[i], color = colors[i], linestyle = linestyles[i])
    axs[1,1].set_title('0.90 quantile of cumul. SubOpt arms', fontsize = 8)
    axs[1,1].legend(loc = 'lower right', fontsize=6)

fig.legend(loc='center left', labels=agents)

fig.tight_layout()
fig.suptitle('SubOpt Metrics from '+str(Rep)+' simulations', fontsize = 10)
fig.subplots_adjust(top=0.88, left=0.25)
plt.savefig(r'file_dir_2')

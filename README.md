# The Asymptotic Randomised Control Algorithm for Contextual Bandits

This is a fork of the TF-Agents library used for the project ‘The Asymptotic Randomised Control Algorithm for Contextual Bandits’. It has an additional implementation of the Asymptotic Randomised Control algorithm for the multi-armed bandit problem, adapted the contextual bandit case. It also has additional support for per arm environments with rewards sampled with a variance that is a function of the chosen observation. For an overview of the implementation and its subsequent performance comparison against LinearUCB and Thompson Sampling algorithms, read the paper [here](https://samuel-howard.github.io/ARC_for_Contextual_bandits.pdf).

In order to implement the ARC algorithm and the new environment, the following modules in TF-Agents have been modified or added:
```
agents/tf_agents/bandits/agents/linear_bandit_agent.py
agents/tf_agents/bandits/agents/lin_ucb_agent.py
agents/tf_agents/bandits/agents/linear_thompson_sampling_agent.py
agents/tf_agents/bandits/agents/lin_arc_agent.py
agents/tf_agents/bandits/policies/linear_bandit_policy.py
agents/tf_agents/bandits/policies/arc_policy.py
agents/tf_agents/bandits/metrics/tf_metrics.py
```

The scripts used for the examples in the research summary are:
```
agents/tf_agents/bandits/agents/examples/ARC_examples/parallel_sspe_general_case.py
agents/tf_agents/bandits/agents/examples/ARC_examples/parallel_per_arm_general_case.py
agents/tf_agents/bandits/agents/examples/ARC_examples/parallel_per_arm_with_var_1.py
agents/tf_agents/bandits/agents/examples/ARC_examples/parallel_per_arm_with_var_2.py
agents/tf_agents/bandits/agents/examples/ARC_examples/parallel_graphical_bandit_1.py
agents/tf_agents/bandits/agents/examples/ARC_examples/parallel_graphical_bandit_2.py
```

To run the code yourself, install the library to a new environment (here I describe the process using Anaconda3), and then install the relevant libraries. The commands I used in the Anaconda prompt are as follows, although these may be different depending on the libraries already installed:

```
conda create --name arc
activate arc
conda install pip git
pip install tensorflow
pip install git+https://github.com/samuel-howard/agents
pip install pillow
pip install scipy
pip install matplotlib
pip install joblib
pip install networkx
```

Then clone the repository at https://github.com/samuel-howard/agents locally, and in the `arc` environment navigate to the local version of the repository to run the above scripts.

You can adapt the examples used in the research summary to create different bandit problems. The variable `reward_param` is the hidden parameter, called \theta in the ARC literature. The context sampling functions `global_context_sampling_fn` and `per_arm_context_sampling_fn` define how the global contexts and per arm contexts are sampled respectively, and can be changed (ensure both are defined in the per arm environment cases, even if they are not used e.g. in the case when the global dimension is 0). The function `variance_function` can also be changed.

Other variables that can be changed are:
`NUM_ACTIONS` - the number of actions that are available for the algorithm to choose from at each timestep.
`BATCH_SIZE` - the size of the batch used in each parameter update.
`ncpu` - the number of cores to use when running the simulations in parallel.
`Rep` - the number of simulations that are run in total (note that this includes any run in previous simulations - see below)
`num_iterations` - the horizon of the game i.e. the number of times we can play the bandit, denoted T in the ARC literature.

If you would like to use the results of simulations that have already been run (which are stored using the pickle module) then set the variable `use_previous_sims` to `True`. If this is done, ensure that all other parameters are set exactly the same as in the simulations that have already been run.

To change which agents are used, ensure that all agents are defined in the `DEFINE POSSIBLE AGENTS` section, along with the correct names, and are listed in the `possible_agents` list. Then set the names of the agents you would like to run in the `agents` list.

The following in the README from the original TF-Agents library..


# TF-Agents: A reliable, scalable and easy to use TensorFlow library for Contextual Bandits and Reinforcement Learning.

[![PyPI tf-agents](https://badge.fury.io/py/tf-agents.svg)](https://badge.fury.io/py/tf-agents)

[TF-Agents](https://github.com/tensorflow/agents) makes implementing, deploying,
and testing new Bandits and RL algorithms easier. It provides well tested and
modular components that can be modified and extended. It enables fast code
iteration, with good test integration and benchmarking.

To get started, we recommend checking out one of our Colab tutorials. If you
need an intro to RL (or a quick recap),
[start here](docs/tutorials/0_intro_rl.ipynb). Otherwise, check out our
[DQN tutorial](docs/tutorials/1_dqn_tutorial.ipynb) to get an agent up and
running in the Cartpole environment. API documentation for the current stable
release is on
[tensorflow.org](https://www.tensorflow.org/agents/api_docs/python/tf_agents).

TF-Agents is under active development and interfaces may change at any time.
Feedback and comments are welcome.

## Table of contents

<a href='#Agents'>Agents</a><br>
<a href='#Tutorials'>Tutorials</a><br>
<a href='#Multi-Armed Bandits'>Multi-Armed Bandits</a><br>
<a href='#Examples'>Examples</a><br>
<a href='#Installation'>Installation</a><br>
<a href='#Contributing'>Contributing</a><br>
<a href='#Releases'>Releases</a><br>
<a href='#Principles'>Principles</a><br>
<a href='#Contributors'>Contributors</a><br>
<a href='#Citation'>Citation</a><br>
<a href='#Disclaimer'>Disclaimer</a><br>

<a id='Agents'></a>

## Agents

In TF-Agents, the core elements of RL algorithms are implemented as `Agents`. An
agent encompasses two main responsibilities: defining a Policy to interact with
the Environment, and how to learn/train that Policy from collected experience.

Currently the following algorithms are available under TF-Agents:

*   [DQN: __Human level control through deep reinforcement learning__ Mnih et
    al., 2015](https://deepmind.com/research/dqn/)
*   [DDQN: __Deep Reinforcement Learning with Double Q-learning__ Hasselt et
    al., 2015](https://arxiv.org/abs/1509.06461)
*   [DDPG: __Continuous control with deep reinforcement learning__ Lillicrap et
    al., 2015](https://arxiv.org/abs/1509.02971)
*   [TD3: __Addressing Function Approximation Error in Actor-Critic Methods__
    Fujimoto et al., 2018](https://arxiv.org/abs/1802.09477)
*   [REINFORCE: __Simple Statistical Gradient-Following Algorithms for
    Connectionist Reinforcement Learning__ Williams,
    1992](https://www-anw.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf)
*   [PPO: __Proximal Policy Optimization Algorithms__ Schulman et al., 2017](https://arxiv.org/abs/1707.06347)
*   [SAC: __Soft Actor Critic__ Haarnoja et al., 2018](https://arxiv.org/abs/1812.05905)

<a id='Tutorials'></a>

## Tutorials

See [`docs/tutorials/`](docs/tutorials) for tutorials on the major components
provided.

<a id='Multi-Armed Bandits'></a>

## Multi-Armed Bandits

The TF-Agents library contains a comprehensive Multi-Armed Bandits suite,
including Bandits environments and agents. RL agents can also be used on Bandit
environments. There is a tutorial in
[`bandits_tutorial.ipynb`](https://github.com/tensorflow/agents/tree/master/docs/tutorials/bandits_tutorial.ipynb).
and ready-to-run examples in
[`tf_agents/bandits/agents/examples/v2`](https://github.com/tensorflow/agents/tree/master/tf_agents/bandits/agents/examples/v2).

<a id='Examples'></a>

## Examples

End-to-end examples training agents can be found under each agent directory.
e.g.:

*   DQN:
    [`tf_agents/agents/dqn/examples/v2/train_eval.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/agents/dqn/examples/v2/train_eval.py)

<a id='Installation'></a>

## Installation

TF-Agents publishes nightly and stable builds. For a list of releases read the
<a href='#Releases'>Releases</a> section. The commands below cover installing
TF-Agents stable and nightly from [pypi.org](https://pypi.org) as well as from a
GitHub clone.

### Stable

Run the commands below to install the most recent stable release. API
documentation for the release is on
[tensorflow.org](https://www.tensorflow.org/agents/api_docs/python/tf_agents).

```shell
$ pip install --user tf-agents[reverb]

# Use this tag get the matching examples and colabs.
$ git clone https://github.com/tensorflow/agents.git
$ cd agents
$ git checkout v0.8.0
```

If you want to install TF-Agents with versions of Tensorflow or
[Reverb](https://github.com/deepmind/reverb) that are flagged as not compatible
by the pip dependency check, use the following pattern below at your own risk.

```shell
$ pip install --user tensorflow
$ pip install --user dm-reverb
$ pip install --user tf-agents
```

If you want to use TF-Agents with TensorFlow 1.15 or 2.0, install version 0.3.0:

```shell
# Newer versions of tensorflow-probability require newer versions of TensorFlow.
$ pip install tensorflow-probability==0.8.0
$ pip install tf-agents==0.3.0
```

### Nightly

Nightly builds include newer features, but may be less stable than the versioned
releases. The nightly build is pushed as `tf-agents-nightly`. We suggest
installing nightly versions of TensorFlow (`tf-nightly`) and TensorFlow
Probability (`tfp-nightly`) as those are the versions TF-Agents nightly are
tested against.

To install the nightly build version, run the following:

```shell
# `--force-reinstall helps guarantee the right versions.
$ pip install --user --force-reinstall tf-nightly
$ pip install --user --force-reinstall tfp-nightly
$ pip install --user --force-reinstall dm-reverb-nightly

# Installing with the `--upgrade` flag ensures you'll get the latest version.
$ pip install --user --upgrade tf-agents-nightly
```

### From GitHub

After cloning the repository, the dependencies can be installed by running `pip
install -e .[tests]`. TensorFlow needs to be installed independently: `pip
install --user tf-nightly`.

<a id='Contributing'></a>

## Contributing

We're eager to collaborate with you! See [`CONTRIBUTING.md`](CONTRIBUTING.md)
for a guide on how to contribute. This project adheres to TensorFlow's
[code of conduct](CODE_OF_CONDUCT.md). By participating, you are expected to
uphold this code.

<a id='Releases'></a>

## Releases

TF Agents has stable and nightly releases. The nightly releases are often fine
but can have issues due to upstream libraries being in flux. The table below
lists the version(s) of TensorFlow tested with each TF Agents' release to help
users that may be locked into a specific version of TensorFlow. 0.3.0 was the
last release compatible with Python 2.

Release | Branch / Tag                                               | TensorFlow Version
------- | ---------------------------------------------------------- | ------------------
Nightly | [master](https://github.com/tensorflow/agents)             | tf-nightly
0.8.0   | [v0.8.0](https://github.com/tensorflow/agents/tree/v0.8.0) | 2.5.0
0.7.1   | [v0.7.1](https://github.com/tensorflow/agents/tree/v0.7.1) | 2.4.0
0.6.0   | [v0.6.0](https://github.com/tensorflow/agents/tree/v0.6.0) | 2.3.0
0.5.0   | [v0.5.0](https://github.com/tensorflow/agents/tree/v0.5.0) | 2.2.0
0.4.0   | [v0.4.0](https://github.com/tensorflow/agents/tree/v0.4.0) | 2.1.0
0.3.0   | [v0.3.0](https://github.com/tensorflow/agents/tree/v0.3.0) | 1.15.0 and 2.0.0

<a id='Principles'></a>

## Principles

This project adheres to [Google's AI principles](PRINCIPLES.md). By
participating, using or contributing to this project you are expected to adhere
to these principles.


<a id='Contributors'></a>

## Contributors


We would like to recognize the following individuals for their code
contributions, discussions, and other work to make the TF-Agents library.

* James Davidson
* Ethan Holly
* Toby Boyd
* Summer Yue
* Robert Ormandi
* Kuang-Huei Lee
* Alexa Greenberg
* Amir Yazdanbakhsh
* Yao Lu
* Gaurav Jain
* Christof Angermueller
* Mark Daoust
* Adam Wood


<a id='Citation'></a>

## Citation

If you use this code, please cite it as:

```
@misc{TFAgents,
  title = {{TF-Agents}: A library for Reinforcement Learning in TensorFlow},
  author = {Sergio Guadarrama and Anoop Korattikara and Oscar Ramirez and
     Pablo Castro and Ethan Holly and Sam Fishman and Ke Wang and
     Ekaterina Gonina and Neal Wu and Efi Kokiopoulou and Luciano Sbaiz and
     Jamie Smith and Gábor Bartók and Jesse Berent and Chris Harris and
     Vincent Vanhoucke and Eugene Brevdo},
  howpublished = {\url{https://github.com/tensorflow/agents}},
  url = "https://github.com/tensorflow/agents",
  year = 2018,
  note = "[Online; accessed 25-June-2019]"
}
```

<a id='Disclaimer'></a>

## Disclaimer

This is not an official Google product.

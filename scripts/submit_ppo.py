import opensim as osim
from osim.http.client import Client
from osim.env import *
import numpy as np
import argparse

import tensorflow as tf

from baselines.common import set_global_seeds, tf_util as U, zipsame
from baselines import bench, logger

from baselines.pposgd import mlp_policy, pposgd_simple

from baselines.common.mpi_adam import MpiAdam
from baselines.common.mpi_moments import mpi_moments

import os.path as osp
import gym, logging
from gym import utils

from baselines import logger
import sys


# Settings
remote_base = 'http://grader.crowdai.org:1729'

# Command line parameters
parser = argparse.ArgumentParser(description='Submit the result to crowdAI')
parser.add_argument('--token', dest='token', action='store', required=True)
parser.add_argument('--logdir', type=str, default='saves')
parser.add_argument('--agentName', type=str, default='PPO-Agent-96-ELU')
parser.add_argument('--hid_size', type=int, default=96)
parser.add_argument('--num_hid_layers', type=int, default=2)
parser.add_argument('--resume', type=int, default=0)
args = parser.parse_args()

sess = U.single_threaded_session()
sess.__enter__()

def policy_fn(name, ob_space, ac_space):
    return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
        hid_size=args.hid_size, num_hid_layers=args.num_hid_layers)


env = RunEnv(visualize=False)
ob_space = env.observation_space
ac_space = env.action_space

pi = policy_fn("pi", ob_space, ac_space) # Construct network for new policy
oldpi = policy_fn("oldpi", ob_space, ac_space) # Network for old policy
atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
ret = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return

clip_param = 0.2
entcoeff = 0.0
adam_epsilon=1e-5

lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[]) # learning rate multiplier, updated with schedule
clip_param = clip_param * lrmult # Annealed cliping parameter epislon

ob = U.get_placeholder_cached(name="ob")
ac = pi.pdtype.sample_placeholder([None])

kloldnew = oldpi.pd.kl(pi.pd)
ent = pi.pd.entropy()
meankl = U.mean(kloldnew)
meanent = U.mean(ent)
pol_entpen = (-entcoeff) * meanent

ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac)) # pnew / pold
surr1 = ratio * atarg # surrogate from conservative policy iteration
surr2 = U.clip(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg #
pol_surr = - U.mean(tf.minimum(surr1, surr2)) # PPO's pessimistic surrogate (L^CLIP)
vf_loss = U.mean(tf.square(pi.vpred - ret))
total_loss = pol_surr + pol_entpen + vf_loss

losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent]
loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]

var_list = pi.get_trainable_variables()
lossandgrad = U.function([ob, ac, atarg, ret, lrmult], losses + [U.flatgrad(total_loss, var_list)])
adam = MpiAdam(var_list, epsilon=adam_epsilon)

assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
    for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
compute_losses = U.function([ob, ac, atarg, ret, lrmult], losses)

U.initialize()
adam.sync()

#pi = policy_fn("pi", ob_space, ac_space) # Construct network for the trained policy

#ob = U.get_placeholder_cached(name="ob")
#ac = pi.pdtype.sample_placeholder([None])

saver = tf.train.Saver()
if args.resume > 0:
    print("Loading agent num ", args.resume)
    saver.restore(tf.get_default_session(), os.path.join(os.path.abspath(args.logdir), "{}-{}".format(args.agentName, args.resume)))
else:
    print("No weights to load!")

#if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob)

client = Client(remote_base)

# Create environment
observation = client.env_create(args.token)

# Run a single step
#
# The grader runs 3 simulations of at most 1000 steps each. We stop after the last one
episodeReward = 0.0
meanReward = 0.0
while True:
    v = np.array(observation).reshape((env.observation_space.shape[0]))
    action, _ = pi.act(False, v)
    for i in range(len(action)):
        action[i] = 0.5 * action[i] + 0.5
    [observation, reward, done, info] = client.env_step(action.tolist())
    episodeReward += reward
#    print(observation)
    print(reward)
    if done:
        meanReward += episodeReward
        print('Episode reward = ', episodeReward)
        episodeReward = 0.0
        observation = client.env_reset()
        if not observation:
            break

meanReward /= 3.0
print('Mean reward = ', meanReward)

client.submit()

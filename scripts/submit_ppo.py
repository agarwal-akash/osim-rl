import opensim as osim
from osim.http.client import Client
from osim.env import *
import numpy as np
import argparse

import tensorflow as tf

from baselines.common import set_global_seeds, tf_util as U, zipsame
from baselines import bench, logger

from baselines.ppo1 import mlp_policy, pposgd_simple

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
#parser.add_argument('--token', dest='token', action='store', required=True)
parser.add_argument('--logdir', type=str, default='saves_128')
parser.add_argument('--agentName', type=str, default='PPO-128')
parser.add_argument('--hid_size', type=int, default=128)
parser.add_argument('--num_hid_layers', type=int, default=2)
parser.add_argument('--resume', type=int, default=1965)
args = parser.parse_args()

sess = U.single_threaded_session()
sess.__enter__()

def policy_fn(name, ob_space, ac_space):
    return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
        hid_size=args.hid_size, num_hid_layers=args.num_hid_layers)


env = RunEnv(visualize=False)
ob_space = env.observation_space
ac_space = env.action_space

observation_space = ([-math.pi] * 65, [math.pi] * 65)
ob_space = convert_to_gym(observation_space)

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

saver = tf.train.Saver()
if args.resume > 0:
    print("Loading agent num ", args.resume)
    saver.restore(tf.get_default_session(), os.path.join(os.path.abspath(args.logdir), "{}-{}".format(args.agentName, args.resume)))
else:
    print("No weights to load!")

#if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob)

def observation_filter(ob, prevob, prevac):
    lob = len(ob)
    lac = len(prevac)
    observation = np.zeros(lob + lac + 6)
    for i in range(lob):
        observation[i] = ob[i]

    xp = observation[1]
    observation[1] = 0.0
    observation[18] -= xp
    observation[22] -= xp
    observation[24] -= xp
    observation[26] -= xp
    observation[28] -= xp
    observation[30] -= xp
    observation[32] -= xp
    observation[34] -= xp

    idt = 100.0
    observation[0] = (ob[22] - prevob[22]) * idt
    observation[41] = (ob[24] - prevob[24]) * idt
    observation[42] = (ob[26] - prevob[26]) * idt
    observation[43] = (ob[28] - prevob[28]) * idt
    observation[44] = (ob[30] - prevob[30]) * idt
    observation[45] = (ob[32] - prevob[32]) * idt
    observation[46] = (ob[34] - prevob[34]) * idt

    for i in range(lac):
        observation[46 + i + 1] = prevac[i]

    return observation


client = Client(remote_base)

# Create environment
observation = client.env_create('96dce98d36c80beead4d56faa0380d4d') #'ef125dcc4a82b5f162cc7f401c4c58a1') #args.token)

prevac = np.zeros(env.action_space.shape)
prevob = np.copy(np.array(observation).reshape((env.observation_space.shape[0])))
#action = action_filter(np.zeros(env.action_space.shape))

# Run a single step
#
# The grader runs 3 simulations of at most 1000 steps each. We stop after the last one
rewards = []
episodeReward = 0.0
while True:
    v = np.array(observation).reshape((env.observation_space.shape[0]))
    ob = observation_filter(v, prevob, prevac)

    action, _ = pi.act(False, ob)
    for i in range(len(action)):
        action[i] = 0.5 * action[i] + 0.5
    action = np.clip(action, 0.0, 1.0)
#    action = action_filter(action)

    prevob = np.copy(v)
    prevac = np.copy(action)

    [observation, reward, done, info] = client.env_step(action.tolist())

    episodeReward += reward
    print(reward)
#    print(observation)
    if done:
        rewards.append(episodeReward)
        print('Episode reward = ', episodeReward)
        episodeReward = 0.0
        observation = client.env_reset()
        if not observation:
            break

for rew in rewards:
    print(' ', rew)
meanReward = np.mean(rewards)
print('Mean reward = ', meanReward)

client.submit()

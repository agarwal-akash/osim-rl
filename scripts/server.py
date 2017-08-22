# Derived from keras-rl
import opensim as osim
import numpy as np
import sys
import zmq

import numpy as np

from osim.env import *
from osim.http.client import Client

import argparse
import math
import struct

# Command line parameters
parser = argparse.ArgumentParser(description='Train or test neural net motor controller')

parser.add_argument('--visualize', dest='visualize', action='store_true', default=False)
parser.add_argument('--portnum', dest='portnum', action='store', default=10000, type=int)
parser.add_argument('--difficulty', dest='difficulty', action='store', default=2, type=int)
parser.add_argument('--pos-noise', dest='pos_noise', action='store', default=0.0, type=float)
parser.add_argument('--vel-noise', dest='vel_noise', action='store', default=0.0, type=float)
args = parser.parse_args()

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:" + str(args.portnum))
print("done connect")

env = RunEnv(visualize=args.visualize)

USE_BINARY_PROTO        = True
USE_COALESCED_ARRAYS    = False

if (USE_COALESCED_ARRAYS):
    print("Calesced array not support yet")
    exit(0)
num_agents = 1

def observation_filter(observation):
    return observation

def action_filter(action):
    return action

def reward_filter(observation, action, reward):
    return reward

# need to change if
observation = observation_filter(env.reset(difficulty=args.difficulty,pos_noise=args.pos_noise,vel_noise=args.vel_noise))
action = action_filter(np.zeros(env.action_space.shape))
numo = len(observation)
numa = len(action)

print("numo = " + str(numo) + " numa = " + str(numa))
sumreward = 0
numsteps = 0
while True:
    message = socket.recv()
    off = 0
    if USE_BINARY_PROTO:
        cmd = struct.unpack_from('@B', message, offset=off)[0]
        off+=1
        req = bytearray()
    else:
        vals = message.split()
        cmd = vals[0]
        req = ""

    if cmd == 99:
        if USE_BINARY_PROTO:
            req = struct.pack('=ii', numo,numa)
        else:
            req = str(numo) + " " + str(numa)
    elif cmd == 0:
        # reset
        sumreward = 0
        numsteps = 0
        observation = observation_filter(env.reset(difficulty=args.difficulty, pos_noise=args.pos_noise, vel_noise=args.vel_noise))
        if USE_BINARY_PROTO:
            for i in range(numo):
                req.extend(struct.pack('=f', observation[i]))
        else:
            req = str(observation[0])
            for i in range(numo)-1:
                req += " " + str(observation[i+1])
    elif cmd == 1:
        # step
        action = np.zeros([numa])
        if USE_BINARY_PROTO:
            for i in range(numa):
                action[i] = struct.unpack_from('@f', message, offset=off)[0]
                off += 4
        else:
            for i in range(numa):
                action[i] = vals[i+1]

        # Remap action to 0..1
        for i in range(numa):
            action[i] = 0.5*action[i] + 0.5
        action = action_filter(action)
        observation, reward, done, info = env.step(action)
        observation = observation_filter(observation)
        reward = reward_filter(observation, action, reward)
        sumreward += reward
        numsteps += 1
        if (done):
            donei = 1
            print("Episode steps = " + str(numsteps) + " reward = " + str(sumreward))
        else:
            donei = 0
        if USE_BINARY_PROTO:
            for i in range(numo):
                req.extend(struct.pack('=f', observation[i]))
            req.extend(struct.pack('=f', reward))
            req.extend(struct.pack('B', donei))
        else:
            req = str(observation[0])
            for i in range(numo)-1:
                req += " " + str(observation[i+1])
            req += " " + str(reward) + " " + str(donei)

    if USE_BINARY_PROTO:
        socket.send(req)
    else:
        socket.send_string(req)



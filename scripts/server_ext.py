# Derived from keras-rl
import opensim as osim
import numpy as np
import sys
import zmq
import time

import numpy as np

from osim.env import *
from osim.http.client import Client

import argparse
import math
import struct

# Command line parameters
parser = argparse.ArgumentParser(description='Train or test neural net motor controller')

parser.add_argument('--visualize', dest='visualize', action='store_true', default=False)
parser.add_argument('--portnum', dest='portnum', action='store', default=5000, type=int)
parser.add_argument('--difficulty', dest='difficulty', action='store', default=2, type=int)
parser.add_argument('--pos-noise', dest='pos_noise', action='store', default=0.2, type=float)
parser.add_argument('--vel-noise', dest='vel_noise', action='store', default=0.1, type=float)
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

def observation_filter(ob, prevob):
    lob = len(ob)
    observation = np.zeros(lob+6)
    for i in range(lob):
        observation[i] = ob[i];

    xp = observation[1]
    observation[1] = 0
    observation[18] -= xp
    observation[22] -= xp
    observation[24] -= xp
    observation[26] -= xp
    observation[28] -= xp
    observation[30] -= xp
    observation[32] -= xp
    observation[34] -= xp
    idt = 100.0
    observation[0] = (ob[22]-prevob[22])*idt
    observation[41] = (ob[24] - prevob[24]) * idt
    observation[42] = (ob[26] - prevob[26]) * idt
    observation[43] = (ob[28] - prevob[28]) * idt
    observation[44] = (ob[30] - prevob[30]) * idt
    observation[45] = (ob[32] - prevob[32]) * idt
    observation[46] = (ob[34] - prevob[34]) * idt

    return observation
def action_filter(action):
    return action

def reward_filter(observation, action, reward):
    return reward

# need to change if
ob = env.reset(difficulty=args.difficulty, pos_noise=args.pos_noise, vel_noise=args.vel_noise)
prevob = np.copy(ob)
observation = observation_filter(ob, prevob)
action = action_filter(np.zeros(env.action_space.shape))
numo = len(observation)
numa = len(action)

print("numo = " + str(numo) + " numa = " + str(numa))
sumreward = 0
numsteps = 0
start_time = time.time()
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
        if numsteps > 0:
            print("Average time per step is "+str((time.time()-start_time)/numsteps))
        start_time = time.time()
        sumreward = 0
        numsteps = 0
        ob = env.reset(difficulty=args.difficulty, pos_noise=args.pos_noise, vel_noise=args.vel_noise)
        prevob = np.copy(ob)
        observation = observation_filter(ob,prevob)
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
        action = np.clip(action, 0.0, 1.0)
        action = action_filter(action)
        observation, reward, done, info = env.step(action)
        ob = observation
        observation = observation_filter(ob, prevob)
        prevob = ob
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



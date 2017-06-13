import copy
from collections import deque
from collections import namedtuple
from random import random, sample

import cv2
import gym
import gym_ple
import numpy as np
import pylab
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import transforms as T
from torchvision.transforms import ToTensor

GAME_NAME = 'FlappyBird-v0'  # only Pygames are supported
INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.05
EXPLORATION_STEPS = 1000000
TARGET_UPDATE_INTERVAL = 10000
BATCH_SIZE = 32


class ReplayMemory(object):
    def __init__(self, capacity=50000):
        self.capacity = capacity
        self.memory = deque(maxlen=self.capacity)
        self.Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))

    def put(self, state, action, reward, next_state):
        # state = torch.FloatTensor(state)
        # action = torch.FloatTensor(action)
        # reward = torch.FloatTensor([reward])
        # next_state = torch.FloatTensor(next_state)
        transition = self.Transition(state=state, action=action, reward=reward, next_state=next_state)
        self.memory.append(transition)

    def sample(self, batch_size):
        transitions = sample(self.memory, batch_size)
        return self.Transition(*(zip(*transitions)))

    def size(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, n_action):
        super(DQN, self).__init__()
        self.n_action = n_action

        self.conv1 = nn.Conv3d(3, 16, kernel_size=5, stride=1, padding=1)  # (In Channel, Out Channel, ...)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=4, stride=1, padding=1)
        self.conv3 = nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1)

        self.bn1 = nn.BatchNorm3d(16)
        self.bn2 = nn.BatchNorm3d(32)
        self.bn3 = nn.BatchNorm3d(32)

        self.affine1 = nn.Linear(209952, 256)
        self.affine2 = nn.Linear(256, self.n_action)

    def forward(self, x):
        h = F.leaky_relu(self.bn1(self.conv1(x)))
        h = F.leaky_relu(self.bn2(self.conv2(h)))
        h = F.leaky_relu(self.bn3(self.conv3(h)))
        h = self.affine1(h.view(h.size(0), -1))
        h = self.affine2(h)
        return h


class Environment(object):
    def __init__(self, game, width=84, height=84):
        self.game = gym.make(game)
        self.width = width
        self.height = height
        self._toTensor = T.Compose([T.ToPILImage(), T.ToTensor()])
        gym_ple

    def play_sample(self, mode: str = 'human'):
        observation = self.game.reset()

        while True:
            screen = self.game.render(mode=mode)
            if mode == 'rgb_array':
                screen = self.preprocess(screen)
            action = self.game.action_space.sample()
            observation, reward, done, info = self.game.step(action)
            if done:
                break
        self.game.close()

    def preprocess(self, screen):
        preprocessed: np.array = cv2.resize(screen, (self.height, self.width))  # 84 * 84 로 변경
        preprocessed: np.array = preprocessed.transpose((2, 0, 1))  # (C, W, H) 로 변경
        preprocessed: np.array = preprocessed.astype('float32') / 255.
        return preprocessed

    def init(self):
        """
        @return observation
        """
        return self.game.reset()

    def get_screen(self):
        screen = self.game.render('rgb_array')
        screen = self.preprocess(screen)
        return screen

    def step(self, action: torch.LongTensor):
        action = action.cpu().max()
        observation, reward, done, info = self.game.step(action)
        return observation, reward, done, info

    def reset(self):
        """
        :return: observation array
        """
        observation = self.game.reset()
        observation = self.preprocess(observation)
        return observation

    @property
    def action_space(self):
        return self.game.action_space.n


class Agent(object):
    def __init__(self, cuda=True, action_repeat: int = 4):
        # Init
        self.action_repeat: int = action_repeat
        self._state_buffer = deque(maxlen=self.action_repeat)
        self.step = 0

        # Environment
        self.env = Environment(GAME_NAME)

        # DQN Model
        self.dqn: DQN = DQN(self.env.action_space)
        if cuda:
            self.dqn.cuda()

        # DQN Target Model
        self.target: DQN = copy.deepcopy(self.dqn)

        # Optimizer
        self.optimizer = optim.RMSprop(self.dqn.parameters(), lr=0.007)

        # Replay Memory
        self.replay = ReplayMemory()

        # Epsilon
        self.epsilon = INITIAL_EPSILON
        self.epsilon_decay = (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORATION_STEPS  # 9.499999999999999e-07

    def clone_dqn(self, dqn):
        return copy.deepcopy(dqn)

    def select_action(self, states: np.array) -> torch.LongTensor:
        # Decrease epsilon value
        self.epsilon -= self.epsilon_decay

        if self.epsilon > random():
            # Random Action
            action = np.zeros(self.env.game.action_space.n, dtype='int64')
            action[self.env.game.action_space.sample()] = 1
            action = torch.LongTensor(action)
            return action

        # max(dimension) 이 들어가며 tuple을 return값으로 내놓는다.
        # tuple안에는 (FloatTensor, LongTensor)가 있으며
        # FloatTensor는 가장 큰 값
        # LongTensor에는 가장 큰 값의 index가 있다.
        states = states.reshape(1, 3, self.action_repeat, self.env.width, self.env.height)
        states_variable: Variable = Variable(torch.FloatTensor(states).cuda())
        action = self.dqn(states_variable).data.max(1)
        return action

    def get_initial_states(self):
        state = self.env.reset()
        state = self.env.get_screen()
        states = np.stack([state for _ in range(self.action_repeat)], axis=0)

        self._state_buffer = deque(maxlen=self.action_repeat)
        for _ in range(self.action_repeat):
            self._state_buffer.append(state)
        return states

    def add_state(self, state):
        self._state_buffer.append(state)

    def recent_states(self):
        return np.array(self._state_buffer)

    def train(self, mode: str = 'rgb_array'):

        # Initial States
        states = self.get_initial_states()
        self.step: int = 0

        while True:
            # Get Action
            action = self.select_action(states)

            # step 에서 나온 observation은 버림
            observation, reward, done, info = self.env.step(action)
            next_state = self.env.get_screen()

            self.add_state(next_state)

            # Store the infomation in Replay Memory
            next_states = self.recent_states()
            self.replay.put(states, action, reward, next_states)

            # Optimize
            if self.step > BATCH_SIZE:
                self.optimize()
                break

            if done:
                break

            # Increase step
            self.step += 1

    def optimize(self):
        transitions = self.replay.sample(BATCH_SIZE)

        state_batch = Variable(torch.FloatTensor(np.array(transitions.state, dtype='float32')).cuda())
        action_batch = Variable(torch.LongTensor(np.array(transitions.action, dtype='int64')).cuda())
        reward_batch = Variable(torch.FloatTensor(np.array(transitions.reward, dtype='float32')).cuda())

        state_batch = state_batch.view([BATCH_SIZE, 3, self.action_repeat, self.env.width, self.env.height])

        print(action_batch)
        print(self.dqn(state_batch)[0])
        print(self.dqn(state_batch).gather(1, action_batch))

    def imshow(self, sample_image: np.array, transpose=True):
        if transpose:
            sample_image = sample_image.transpose((1, 2, 0))
        pylab.imshow(sample_image)
        pylab.show()


def main():
    # env = Environment(GAME_NAME)
    # env.play_sample()

    agent = Agent()
    agent.train()


if __name__ == '__main__':
    main()

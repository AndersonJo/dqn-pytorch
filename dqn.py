from random import random

import cv2
import gym
import gym_ple
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import transforms as T
import copy

GAME_NAME = 'FlappyBird-v0'  # only Pygames are supported
INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.05
EXPLORATION_STEPS = 1000000


class ReplayMemory(object):
    pass


class DQN(nn.Module):
    def __init__(self, n_action):
        super(DQN, self).__init__()
        self.n_action = n_action

        self.conv1 = nn.Conv2d(3, 16, kernel_size=20, stride=2)  # (In Channel, Out Channel, ...)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=9, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=2)

        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)

        self.affine1 = nn.Linear(512, self.n_action)

    def forward(self, x):
        h = F.leaky_relu(self.bn1(self.conv1(x)))
        h = F.leaky_relu(self.bn2(self.conv2(h)))
        h = F.leaky_relu(self.bn3(self.conv3(h)))
        out = self.affine1(h.view(h.size(0), -1))
        return out


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
        preprocessed = cv2.resize(screen, (self.height, self.width))  # 84 * 84 로 변경
        preprocessed = preprocessed.transpose((2, 0, 1))  # (C, W, H) 로 변경
        return preprocessed

    def init(self):
        """
        @return observation
        """
        return self.game.reset()

    def get_screen(self):
        screen = self.game.render(mode='rgb_array')
        screen = self.preprocess(screen)
        return screen

    def toVariable(self, x):
        return Variable(self._toTensor(x).cuda())

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
    def __init__(self, cuda=True):
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
        self.replay_momory = ReplayMemory()

        # Epsilon
        self.epsilon = INITIAL_EPSILON
        self.epsilon_decay = (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORATION_STEPS  # 9.499999999999999e-07

        print(self.epsilon, self.epsilon_step)

    def clone_dqn(self, dqn):
        return copy.deepcopy(dqn)

    def select_action(self, state):
        rand_value = random()
        if self.epsilon < rand_value:
            pass

    def train(self, mode: str = 'rgb_array'):
        observation = self.env.reset()
        screen = self.env.get_screen()


        # while True:
        # self.select_action(state)


def main():
    # env = Environment(GAME_NAME)
    # env.play_sample()

    agent = Agent()
    agent.train()


if __name__ == '__main__':
    main()

import argparse
import copy
import glob
import logging
import math
import os
import re
import sys
from collections import deque
from collections import namedtuple
from random import random, sample

import cv2
import gym
import gym_ple
import numpy as np
import pylab
import torch
from gym.wrappers import Monitor
from scipy.misc import toimage
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import transforms as T

# Support Games
# 'FlappyBird-v0'
# 'MonsterKon-v0'

# Training
BATCH_SIZE = 32

# Replay Memory
REPLAY_MEMORY = 50000

# Epsilon
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 100000

# LSTM Memory
LSTM_MEMORY = 128

# ETC Options
TARGET_UPDATE_INTERVAL = 1000
CHECKPOINT_INTERVAL = 5000
PLAY_INTERVAL = 900
PLAY_REPEAT = 1
LEARNING_RATE = 0.0001

parser = argparse.ArgumentParser(description='DQN Configuration')
parser.add_argument('--model', default='dqn', type=str, help='forcefully set step')
parser.add_argument('--step', default=None, type=int, help='forcefully set step')
parser.add_argument('--best', default=None, type=int, help='forcefully set best')
parser.add_argument('--load_latest', dest='load_latest', action='store_true', help='load latest checkpoint')
parser.add_argument('--no_load_latest', dest='load_latest', action='store_false', help='train from the scrach')
parser.add_argument('--checkpoint', default=None, type=str, help='specify the checkpoint file name')
parser.add_argument('--mode', dest='mode', default='play', type=str, help='[play, train]')
parser.add_argument('--game', default='FlappyBird-v0', type=str, help='only Pygames are supported')
parser.add_argument('--clip', dest='clip', action='store_true', help='clipping the delta between -1 and 1')
parser.add_argument('--noclip', dest='clip', action='store_false', help='not clipping the delta')
parser.add_argument('--skip_action', default=4, type=int, help='Skipping actions')
parser.add_argument('--record', dest='record', action='store_true', help='Record playing a game')
parser.add_argument('--inspect', dest='inspect', action='store_true', help='Inspect CNN')
parser.add_argument('--seed', default=111, type=int, help='random seed')
parser.set_defaults(clip=True, load_latest=True, record=False, inspect=False)
parser: argparse.Namespace = parser.parse_args()

# Random Seed
torch.manual_seed(parser.seed)
torch.cuda.manual_seed(parser.seed)
np.random.seed(parser.seed)

# Logging
logger = logging.getLogger('DQN')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(message)s')

file_handler = logging.FileHandler(f'dqn_{parser.model}.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


class ReplayMemory(object):
    def __init__(self, capacity=REPLAY_MEMORY):
        self.capacity = capacity
        self.memory = deque(maxlen=self.capacity)
        self.Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))
        self._available = False

    def put(self, state: np.array, action: torch.LongTensor, reward: np.array, next_state: np.array):
        """
        저장시 모두 Torch Tensor로 변경해준다음에 저장을 합니다.
        action은 select_action()함수에서부터 LongTensor로 리턴해주기 때문에,
        여기서 변경해줄필요는 없음
        """
        state = torch.FloatTensor(state)
        reward = torch.FloatTensor([reward])
        if next_state is not None:
            next_state = torch.FloatTensor(next_state)
        transition = self.Transition(state=state, action=action, reward=reward, next_state=next_state)
        self.memory.append(transition)

    def sample(self, batch_size):
        transitions = sample(self.memory, batch_size)
        return self.Transition(*(zip(*transitions)))

    def size(self):
        return len(self.memory)

    def is_available(self):
        if self._available:
            return True

        if len(self.memory) > BATCH_SIZE:
            self._available = True
        return self._available


class DQN(nn.Module):
    def __init__(self, n_action):
        super(DQN, self).__init__()
        self.n_action = n_action

        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=0)  # (In Channel, Out Channel, ...)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)

        self.affine1 = nn.Linear(3136, 512)
        self.affine2 = nn.Linear(512, self.n_action)

    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))

        # print(h.size())
        # print(h.view(h.size(0), -1).size())

        h = F.relu(self.affine1(h.view(h.size(0), -1)))
        h = self.affine2(h)
        return h


class LSTMDQN(nn.Module):
    def __init__(self, n_action):
        super(LSTMDQN, self).__init__()
        self.n_action = n_action

        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=1, padding=1)  # (In Channel, Out Channel, ...)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.lstm = nn.LSTM(16, LSTM_MEMORY, 1)  # (Input, Hidden, Num Layers)

        self.affine1 = nn.Linear(LSTM_MEMORY * 64, 512)
        # self.affine2 = nn.Linear(2048, 512)
        self.affine2 = nn.Linear(512, self.n_action)

    def forward(self, x, hidden_state, cell_state):
        # CNN
        h = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=2))
        h = F.relu(F.max_pool2d(self.conv2(h), kernel_size=2, stride=2))
        h = F.relu(F.max_pool2d(self.conv3(h), kernel_size=2, stride=2))
        h = F.relu(F.max_pool2d(self.conv4(h), kernel_size=2, stride=2))

        # LSTM
        h = h.view(h.size(0), h.size(1), 16)  # (32, 64, 4, 4) -> (32, 64, 16)
        h, (next_hidden_state, next_cell_state) = self.lstm(h, (hidden_state, cell_state))
        h = h.view(h.size(0), -1)  # (32, 64, 256) -> (32, 16348)

        # Fully Connected Layers
        h = F.relu(self.affine1(h.view(h.size(0), -1)))
        # h = F.relu(self.affine2(h.view(h.size(0), -1)))
        h = self.affine2(h)
        return h, next_hidden_state, next_cell_state

    def init_states(self) -> [Variable, Variable]:
        hidden_state = Variable(torch.zeros(1, 64, LSTM_MEMORY).cuda())
        cell_state = Variable(torch.zeros(1, 64, LSTM_MEMORY).cuda())
        return hidden_state, cell_state

    def reset_states(self, hidden_state, cell_state):
        hidden_state[:, :, :] = 0
        cell_state[:, :, :] = 0
        return hidden_state.detach(), cell_state.detach()


class Environment(object):
    def __init__(self, game, record=False, width=84, height=84, seed=0):
        self.game = gym.make(game)
        self.game.seed(seed)

        if record:
            self.game = Monitor(self.game, './video', force=True)

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
        preprocessed = np.dot(preprocessed[..., :3], [0.299, 0.587, 0.114])  # Gray scale 로 변경
        # preprocessed: np.array = preprocessed.transpose((2, 0, 1))  # (C, W, H) 로 변경
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

    def step(self, action: int):
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
    def __init__(self, args: argparse.Namespace, cuda=True, action_repeat: int = 4):
        # Init
        self.clip: bool = args.clip
        self.seed: int = args.seed
        self.action_repeat: int = action_repeat
        self.frame_skipping: int = args.skip_action
        self._state_buffer = deque(maxlen=self.action_repeat)
        self.step = 0
        self.best_score = args.best or -10000
        self.best_count = 0

        self._play_steps = deque(maxlen=5)

        # Environment
        self.env = Environment(args.game, record=args.record, seed=self.seed)

        # DQN Model
        self.dqn_hidden_state = self.dqn_cell_state = None
        self.target_hidden_state = self.target_cell_state = None

        self.mode: str = args.model.lower()
        if self.mode == 'dqn':
            self.dqn: DQN = DQN(self.env.action_space)
        elif self.mode == 'lstm':
            self.dqn: LSTMDQN = LSTMDQN(self.env.action_space)

            # For Optimization
            self.dqn_hidden_state, self.dqn_cell_state = self.dqn.init_states()
            self.target_hidden_state, self.target_cell_state = self.dqn.init_states()

            # For Training Play
            self.train_hidden_state, self.train_cell_state = self.dqn.init_states()

            # For Validation Play
            self.test_hidden_state, self.test_cell_state = self.dqn.init_states()

        if cuda:
            self.dqn.cuda()

        # DQN Target Model
        self.target: DQN = copy.deepcopy(self.dqn)

        # Optimizer
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=LEARNING_RATE)

        # Replay Memory
        self.replay = ReplayMemory()

        # Epsilon
        self.epsilon = EPSILON_START

    def select_action(self, states: np.array) -> tuple:
        """
        :param states: 게임화면
        :return: LongTensor (int64) 값이며, [[index]] 이런 형태를 갖고 있다.
        추후 gather와 함께 쓰기 위해서 index값이 필요하다
        """
        # Decrease epsilon value
        self.epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * \
                                     math.exp(-1. * self.step / EPSILON_DECAY)

        if self.epsilon > random():
            # Random Action
            sample_action = self.env.game.action_space.sample()
            action = torch.LongTensor([[sample_action]])
            return action

        # max(dimension) 이 들어가며 tuple을 return값으로 내놓는다.
        # tuple안에는 (FloatTensor, LongTensor)가 있으며
        # FloatTensor는 가장 큰 값
        # LongTensor에는 가장 큰 값의 index가 있다.
        states = states.reshape(1, self.action_repeat, self.env.width, self.env.height)
        states_variable: Variable = Variable(torch.FloatTensor(states).cuda())

        if self.mode == 'dqn':
            states_variable.volatile = True
            action = self.dqn(states_variable).data.cpu().max(1)[1]
        elif self.mode == 'lstm':
            action, self.dqn_hidden_state, self.dqn_cell_state = \
                self.dqn(states_variable, self.train_hidden_state, self.train_cell_state)
            action = action.data.cpu().max(1)[1]

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

    def train(self, gamma: float = 0.99, mode: str = 'rgb_array'):
        # Initial States
        reward_sum = 0.
        q_mean = [0., 0.]
        target_mean = [0., 0.]

        while True:
            # Init LSTM States
            if self.mode == 'lstm':
                # For Training
                self.train_hidden_state, self.train_cell_state = self.dqn.reset_states(self.train_hidden_state,
                                                                                       self.train_cell_state)

            states = self.get_initial_states()
            losses = []
            checkpoint_flag = False
            target_update_flag = False
            play_flag = False
            play_steps = 0
            real_play_count = 0
            real_score = 0

            reward = 0
            done = False
            while True:
                # Get Action
                action: torch.LongTensor = self.select_action(states)

                for _ in range(self.frame_skipping):
                    # step 에서 나온 observation은 버림
                    observation, reward, done, info = self.env.step(action[0, 0])
                    next_state = self.env.get_screen()
                    self.add_state(next_state)

                    if done:
                        break

                # Store the infomation in Replay Memory
                next_states = self.recent_states()
                if done:
                    self.replay.put(states, action, reward, None)
                else:
                    self.replay.put(states, action, reward, next_states)

                # Change States
                states = next_states

                # Optimize
                if self.replay.is_available():
                    loss, reward_sum, q_mean, target_mean = self.optimize(gamma)
                    losses.append(loss[0])

                if done:
                    break

                # Increase step
                self.step += 1
                play_steps += 1

                # Target Update
                if self.step % TARGET_UPDATE_INTERVAL == 0:
                    self._target_update()
                    target_update_flag = True

                # Checkpoint
                # if self.step % CHECKPOINT_INTERVAL == 0:
                #     self.save_checkpoint(filename=f'dqn_checkpoints/chkpoint_{self.mode}_{self.step}.pth.tar')
                #     checkpoint_flag = True

                # Play
                if self.step % PLAY_INTERVAL == 0:
                    play_flag = True

                    scores = []
                    counts = []
                    for _ in range(PLAY_REPEAT):
                        score, real_play_count = self.play(logging=False, human=False)
                        scores.append(score)
                        counts.append(real_play_count)
                        logger.debug(f'[{self.step}] [Validation] play_score: {score}, play_count: {real_play_count}')
                    real_score = int(np.mean(scores))
                    real_play_count = int(np.mean(counts))

                    if self.best_score <= real_score:
                        self.best_score = real_score
                        self.best_count = real_play_count
                        logger.debug(f'[{self.step}] [CheckPoint] Play: {self.best_score} [Best Play] [checkpoint]')
                        self.save_checkpoint(
                            filename=f'dqn_checkpoints/chkpoint_{self.mode}_{self.best_score}.pth.tar')

            self._play_steps.append(play_steps)

            # Play
            if play_flag:
                play_flag = False
                logger.info(f'[{self.step}] [Validation] mean_score: {real_score}, mean_play_count: {real_play_count}')

            # Logging
            mean_loss = np.mean(losses)
            target_update_msg = '  [target updated]' if target_update_flag else ''
            # save_msg = '  [checkpoint!]' if checkpoint_flag else ''
            logger.info(f'[{self.step}] Loss:{mean_loss:<8.4} Play:{play_steps:<3}  '  # AvgPlay:{self.play_step:<4.3}
                        f'RewardSum:{reward_sum:<3} Q:[{q_mean[0]:<6.4}, {q_mean[1]:<6.4}] '
                        f'T:[{target_mean[0]:<6.4}, {target_mean[1]:<6.4}] '
                        f'Epsilon:{self.epsilon:<6.4}{target_update_msg}')

    def optimize(self, gamma: float):
        if self.mode == 'lstm':
            # For Optimization
            self.dqn_hidden_state, self.dqn_cell_state = self.dqn.reset_states(self.dqn_hidden_state,
                                                                               self.dqn_cell_state)
            self.target_hidden_state, self.target_cell_state = self.dqn.reset_states(self.target_hidden_state,
                                                                                     self.target_cell_state)

        # Get Sample
        transitions = self.replay.sample(BATCH_SIZE)

        # Mask
        non_final_mask = torch.ByteTensor(list(map(lambda ns: ns is not None, transitions.next_state))).cuda()
        final_mask = 1 - non_final_mask

        state_batch: Variable = Variable(torch.cat(transitions.state).cuda())
        action_batch: Variable = Variable(torch.cat(transitions.action).cuda())
        reward_batch: Variable = Variable(torch.cat(transitions.reward).cuda())
        non_final_next_state_batch = Variable(torch.cat([ns for ns in transitions.next_state if ns is not None]).cuda())
        non_final_next_state_batch.volatile = True

        # Reshape States and Next States
        state_batch = state_batch.view([BATCH_SIZE, self.action_repeat, self.env.width, self.env.height])
        non_final_next_state_batch = non_final_next_state_batch.view(
            [-1, self.action_repeat, self.env.width, self.env.height])
        non_final_next_state_batch.volatile = True

        # Clipping Reward between -2 and 2
        reward_batch.data.clamp_(-1, 1)

        # Predict by DQN Model
        if self.mode == 'dqn':
            q_pred = self.dqn(state_batch)
        elif self.mode == 'lstm':
            q_pred, self.dqn_hidden_state, self.dqn_cell_state = self.dqn(state_batch, self.dqn_hidden_state,
                                                                          self.dqn_cell_state)

        q_values = q_pred.gather(1, action_batch)

        # Predict by Target Model
        target_values = Variable(torch.zeros(BATCH_SIZE, 1).cuda())
        if self.mode == 'dqn':
            target_pred = self.target(non_final_next_state_batch)
        elif self.mode == 'lstm':
            target_pred, self.target_hidden_state, self.target_cell_state = self.target(non_final_next_state_batch,
                                                                                        self.target_hidden_state,
                                                                                        self.target_cell_state)

        target_values[non_final_mask] = reward_batch[non_final_mask] + target_pred.max(1)[0] * gamma
        target_values[final_mask] = reward_batch[final_mask].detach()

        loss = F.smooth_l1_loss(q_values, target_values)

        # loss = torch.mean((target_values - q_values) ** 2)
        self.optimizer.zero_grad()
        loss.backward(retain_variables=True)

        if self.clip:
            for param in self.dqn.parameters():
                param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        reward_score = int(torch.sum(reward_batch).data.cpu().numpy()[0])
        q_mean = torch.sum(q_pred, 0).data.cpu().numpy()[0]
        target_mean = torch.sum(target_pred, 0).data.cpu().numpy()[0]

        return loss.data.cpu().numpy(), reward_score, q_mean, target_mean

    def _target_update(self):
        self.target = copy.deepcopy(self.dqn)

    def save_checkpoint(self, filename='dqn_checkpoints/checkpoint.pth.tar'):
        dirpath = os.path.dirname(filename)

        if not os.path.exists(dirpath):
            os.mkdir(dirpath)

        checkpoint = {
            'dqn': self.dqn.state_dict(),
            'target': self.target.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'step': self.step,
            'best': self.best_score,
            'best_count': self.best_count
        }
        torch.save(checkpoint, filename)

    def load_checkpoint(self, filename='dqn_checkpoints/checkpoint.pth.tar', epsilon=None):
        checkpoint = torch.load(filename)
        self.dqn.load_state_dict(checkpoint['dqn'])
        self.target.load_state_dict(checkpoint['target'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.step = checkpoint['step']
        self.best_score = self.best_score or checkpoint['best']
        self.best_count = checkpoint['best_count']

    def load_latest_checkpoint(self, epsilon=None):
        r = re.compile('chkpoint_(dqn|lstm)_(?P<number>-?\d+)\.pth\.tar$')

        files = glob.glob(f'dqn_checkpoints/chkpoint_{self.mode}_*.pth.tar')

        if files:
            files = list(map(lambda x: [int(r.search(x).group('number')), x], files))
            files = sorted(files, key=lambda x: x[0])
            latest_file = files[-1][1]
            self.load_checkpoint(latest_file, epsilon=epsilon)
            print(f'latest checkpoint has been loaded - {latest_file}')
        else:
            print('no latest checkpoint')

    def play(self, logging=True, human=True):
        observation = self.env.game.reset()
        states = self.get_initial_states()
        count = 0
        total_score = 0

        self.env.game.seed(self.seed)

        if self.mode == 'lstm':
            self.test_hidden_state, self.test_cell_state = self.dqn.reset_states(self.test_hidden_state,
                                                                                 self.test_cell_state)

        while True:
            # screen = self.env.game.render(mode='human')

            states = states.reshape(1, self.action_repeat, self.env.width, self.env.height)
            states_variable: Variable = Variable(torch.FloatTensor(states).cuda())

            if self.mode == 'dqn':
                dqn_pred = self.dqn(states_variable)
            elif self.mode == 'lstm':
                dqn_pred, self.dqn_hidden_state, self.dqn_cell_state = \
                    self.dqn(states_variable, self.test_hidden_state, self.test_cell_state)

            action = dqn_pred.data.cpu().max(1)[1][0, 0]

            for _ in range(self.frame_skipping):
                if human:
                    screen = self.env.game.render(mode='human')
                observation, reward, done, info = self.env.step(action)
                # States <- Next States
                next_state = self.env.get_screen()
                self.add_state(next_state)
                states = self.recent_states()

                total_score += reward

                if done:
                    break

            # Logging
            count += 1
            if logging:
                action_dist = torch.sum(dqn_pred, 0).data.cpu().numpy()[0]
                print(f'[{count}] action:{action} {action_dist}, reward:{reward}')

            if done:
                break
        self.env.game.close()
        return total_score, count

    def inspect(self):
        print(dir(self.dqn.conv1))

        for param in list(self.dqn.parameters()):
            print(param.size())

        print(self.dqn.conv2.kernel_size)
        print(self.dqn.conv3.kernel_size)
        print(self.dqn.conv4.kernel_size)
        print(self.dqn.conv5.kernel_size)

    @property
    def play_step(self):
        return np.nan_to_num(np.mean(self._play_steps))

    def _sum_params(self, model):
        return np.sum([torch.sum(p).data[0] for p in model.parameters()])

    def imshow(self, sample_image: np.array, transpose=False):
        if transpose:
            sample_image = sample_image.transpose((1, 2, 0))
        pylab.imshow(sample_image, cmap='gray')
        pylab.show()

    def toimage(self, image: np.array, name: str):
        toimage(image, cmin=0, cmax=255).save(name)


def main(parser):
    agent = Agent(parser)
    if parser.load_latest and not parser.checkpoint:
        agent.load_latest_checkpoint()
    elif parser.checkpoint:
        agent.load_checkpoint(parser.checkpoint)

    if parser.mode.lower() == 'play':
        agent.play()
    elif parser.mode.lower() == 'train':
        agent.train()
    elif parser.mode.lower() == 'inspect':
        agent.inspect()


if __name__ == '__main__':
    main(parser)

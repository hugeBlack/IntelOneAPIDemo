import gymnasium as gym
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import deque, namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 引入oneAPI中的Intel Extension for Pytorch
import intel_extension_for_pytorch as ipex

batchSize = 128
minEpsilon = 0.05
dEpsilon = 1e-3
learningRate = 1e-4
gamma = 0.99

env = gym.make("CartPole-v1", render_mode="rgb_array")

# 选择xpu作为torch运算硬件
device = torch.device("xpu")

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class DQN(nn.Module):
    def __init__(self, stateSize, actionSize):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(stateSize, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, actionSize)

    def forward(self, x):
        # output.shape: batch_size*n_actions, state_action_value
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Agent:
    def __init__(self, stateSize, actionSize):
        self.actionSize = actionSize
        self.stateSize = stateSize
        # 一开始policyNN和targetNN参数相同
        self.policyNN = DQN(stateSize, actionSize).to(device)
        self.targetNN = DQN(stateSize, actionSize).to(device)
        self.targetNN.load_state_dict(self.policyNN.state_dict())
        # AdamW优化器
        self.optimiser = optim.AdamW(self.policyNN.parameters(), lr=learningRate, amsgrad=True)

        self.nowEpsilon = 1
        self.memory = ReplayMemory(10000)

    # 用epsilon-greedy选择explore还是exploit，state要是tensor
    def selectAction(self, nowState):
        a = random.random()
        self.nowEpsilon = self.nowEpsilon - dEpsilon if self.nowEpsilon - dEpsilon > minEpsilon else minEpsilon
        if a < self.nowEpsilon:
            # explore
            return random.randint(0, self.actionSize - 1)
        else:
            # exploit
            with torch.no_grad():
                return self.policyNN(nowState).max(1)[1].item()

    def recordTransition(self, state, action, next_state, reward):
        self.memory.push(state, action, next_state, reward)

    def updateQ(self):
        if len(self.memory) < batchSize:
            return
        # 对policyNN(Q)进行更新，隔c次再将targetNN与policyNN同步
        transitions = self.memory.sample(batchSize)
        batch = Transition(*zip(*transitions))

        # 计算目标tensor
        notNoneNextStateMask = [False] * batchSize
        for i, t in enumerate(transitions):  # 筛选出St+1不为None的state的mask
            if t.next_state is not None:
                notNoneNextStateMask[i] = True

        # 所有不为空的nextState组成的列向量
        notNoneNextStates = torch.cat([s for s in batch.next_state if s is not None])
        nextStateReward = torch.zeros(batchSize, device=device)  # fi[i]
        with torch.no_grad():
            nextStateReward[notNoneNextStateMask] = self.targetNN(notNoneNextStates).max(1)[0]

        # 计算reward列向量
        batchReward = torch.cat(batch.reward)
        batchState = torch.cat(batch.state)
        batchAction = torch.tensor(batch.action, device=device).unsqueeze(-1)

        currentQ = self.policyNN(batchState).gather(1, batchAction)
        y = batchReward + (gamma * nextStateReward)

        # 计算loss
        loss = F.huber_loss(currentQ, y.unsqueeze(1))

        # 梯度下降
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        # soft update
        targetArgs = self.targetNN.state_dict()
        policyArgs = self.policyNN.state_dict()
        for key in policyArgs:
            targetArgs[key] = policyArgs[key] * 0.005 + targetArgs[key] * (1 - 0.005)
        self.targetNN.load_state_dict(targetArgs)


episode_durations = []

def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.01)  # pause a bit so that plots are updated


if __name__ == "__main__":
    # 获取action与state数
    actionCount = env.action_space.n
    state, info = env.reset()
    stateCount = len(state)

    max_episodes = 600
    complete_episodes = 0
    finished_flag = False
    agent = Agent(stateCount, actionCount)

    for nowEpisode in range(max_episodes):
        # 训练一轮初始化一次gym
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        nowStep = 0
        while True:
            # 选择一个操作，然后记录4个参数到memory里面，并更新网络参数
            action = agent.selectAction(state)
            observation, reward, terminated, truncated, _ = env.step(action)
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                nextState = None
            else:
                nextState = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            agent.recordTransition(state, action, nextState, reward)

            state = nextState
            agent.updateQ()

            if done:
                episode_durations.append(nowStep + 1)
                print(str(nowEpisode) + "  " + str(nowStep))
                plot_durations()
                break

            nowStep += 1


plot_durations(show_result=True)
plt.ioff()
plt.show()
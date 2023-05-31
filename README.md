# Intel® oneAPI简介

Intel oneAPI 是一个跨行业、开放、基于标准的统一的编程模型，它为跨 CPU、GPU、FPGA、专用加速器的开发者提供统一的体验。

它由一项行业计划和一款英特尔beta产品组成。

oneAPI 开放规范基于行业标准和现有开发者编程模型，广泛适用于不同架构和来自不同供应商的硬件。

oneAPI 行业计划鼓励生态系统内基于oneAPI规范的合作以及兼容 oneAPI的实践。

通过oneAPI，我们可以最大程度地忽略硬件之间的差异，而使用统一的编程模型编写程序，有效地提高我们的开发效率

# 基于Intel® oneAPI 的 Deep Q-Learning 算法实现

## 简介

我们可以基于oneAPI编写、训练、使用神经网络，下方的代码展示了如何使用Intel® oneAPI的pytorch extension编写并训练一个 Deep Q-Learning 网络

本代码是DQN算法的一个简单实现，实现了使用DQN算法训练智能体玩CartPole游戏的代码。

## Intel®Extension for PyTorch

该拓展包拓展了pytorch所支持的硬件设备，不再拘泥于CUDA或CPU运行。其利用了英特尔CPU上的AVX-512矢量神经网络指令（AVX512 VNNI）和英特尔Xe高级矩阵扩展，以及英特尔GPU上的Xe矩阵扩展（XMX）AI引擎对AI训练进行加速。

## 依赖

本项目需要intel pytorch extension、pytorch、matplotlib、gymnasium等库才能运行。

安装：
```shell
pip install matplotlib gymnasium torch intel_extension_for_pytorch
```

## 使用XPU加速运算

```python
# 引入oneAPI中的Intel Extension for Pytorch
import intel_extension_for_pytorch as ipex

# 选择xpu作为torch运算硬件
device = torch.device("xpu")
```

## 运行

本项目不需要额外配置，直接运行python文件即可。
```shell
python dqn.py
```

运行时，会显示matplot的图表，显示在一个episode中坚持了多少个step

控制台中会显示当前episode的结果。

## 实验结果

实验结果如图所示：

![image](/Figure_1.png)

可以发现，AI很好地掌握了杆平衡的方法，通过Intel® oneAPI训练所需的时间也较纯CPU训练短。

## 完整代码

```python
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

```
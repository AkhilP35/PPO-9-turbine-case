import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import numpy as np


# -----------------------------------
# Utilities: Buffer & Normalization
# -----------------------------------

class ReplayBuffer:
    def __init__(self, args):
        self.s = np.zeros((args.batch_size, args.state_dim))
        self.a = np.zeros((args.batch_size, args.action_dim))
        self.a_logprob = np.zeros((args.batch_size, args.action_dim))
        self.r = np.zeros((args.batch_size, 1))
        self.s_ = np.zeros((args.batch_size, args.state_dim))
        self.dw = np.zeros((args.batch_size, 1))
        self.done = np.zeros((args.batch_size, 1))
        self.count = 0

    def store(self, s, a, a_logprob, r, s_, dw, done):
        self.s[self.count] = s
        self.a[self.count] = a
        self.a_logprob[self.count] = a_logprob
        self.r[self.count] = r
        self.s_[self.count] = s_
        self.dw[self.count] = dw
        self.done[self.count] = done
        self.count += 1

    def numpy_to_tensor(self):
        s = torch.tensor(self.s, dtype=torch.float)
        a = torch.tensor(self.a, dtype=torch.float)
        a_logprob = torch.tensor(self.a_logprob, dtype=torch.float)
        r = torch.tensor(self.r, dtype=torch.float)
        s_ = torch.tensor(self.s_, dtype=torch.float)
        dw = torch.tensor(self.dw, dtype=torch.float)
        done = torch.tensor(self.done, dtype=torch.float)
        return s, a, a_logprob, r, s_, dw, done


class RunningMeanStd:
    # Dynamically calculates mean and std
    def __init__(self, shape):
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.zeros(shape)

    def update(self, x):
        x = np.array(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n)


class Normalization:
    def __init__(self, shape):
        self.running_ms = RunningMeanStd(shape=shape)

    def __call__(self, x, update=True):
        if update:
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)
        return x


# -----------------------------------
# Neural Networks
# -----------------------------------

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Actor_Gaussian(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_width=128):
        super(Actor_Gaussian, self).__init__()
        self.fc1 = layer_init(nn.Linear(state_dim, hidden_width))
        self.fc2 = layer_init(nn.Linear(hidden_width, hidden_width))
        self.mu = layer_init(nn.Linear(hidden_width, action_dim), std=0.01)
        self.sigma = layer_init(nn.Linear(hidden_width, action_dim), std=0.01)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        mu = torch.tanh(self.mu(x))  # Range [-1, 1]
        sigma = F.softplus(self.sigma(x)) + 0.001  # Ensure positive standard deviation
        return mu, sigma

    def get_dist(self, x):
        mu, sigma = self.forward(x)
        return Normal(mu, sigma)


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_width=128):
        super(Critic, self).__init__()
        self.fc1 = layer_init(nn.Linear(state_dim, hidden_width))
        self.fc2 = layer_init(nn.Linear(hidden_width, hidden_width))
        self.fc3 = layer_init(nn.Linear(hidden_width, 1), std=1.0)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.fc3(x)


# -----------------------------------
# PPO Agent
# -----------------------------------

class PPO_Continuous:
    def __init__(self, args):
        self.args = args
        self.actor = Actor_Gaussian(args.state_dim, args.action_dim, args.hidden_width)
        self.critic = Critic(args.state_dim, args.hidden_width)

        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=args.lr_a, eps=1e-5)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=args.lr_c, eps=1e-5)

    def choose_action(self, s):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        with torch.no_grad():
            dist = self.actor.get_dist(s)
            a = dist.sample()
            a = torch.clamp(a, -1.0, 1.0)  # Clip internal action to valid range
            a_logprob = dist.log_prob(a)
        return a.numpy().flatten(), a_logprob.numpy().flatten()

    def update(self, replay_buffer, total_steps):
        s, a, a_logprob, r, s_, dw, done = replay_buffer.numpy_to_tensor()

        # Calculate Advantage using GAE
        adv = []
        gae = 0
        with torch.no_grad():
            vs = self.critic(s)
            vs_ = self.critic(s_)
            deltas = r + self.args.gamma * (1.0 - dw) * vs_ - vs
            for delta, d in zip(reversed(deltas.flatten().numpy()), reversed(done.flatten().numpy())):
                gae = delta + self.args.gamma * self.args.lamda * gae * (1.0 - d)
                adv.insert(0, gae)
            adv = torch.tensor(adv, dtype=torch.float).view(-1, 1)
            v_target = adv + vs

            # Advantage Normalization
            adv = ((adv - adv.mean()) / (adv.std() + 1e-5))

        # Optimize for K epochs
        for _ in range(self.args.K_epochs):
            for index in BatchSampler(SubsetRandomSampler(range(self.args.batch_size)), self.args.mini_batch_size,
                                      False):
                dist_now = self.actor.get_dist(s[index])
                dist_entropy = dist_now.entropy().sum(1, keepdim=True)
                a_logprob_now = dist_now.log_prob(a[index])

                # Ratio
                ratios = torch.exp(a_logprob_now.sum(1, keepdim=True) - a_logprob[index].sum(1, keepdim=True))

                # Surrogate Loss
                surr1 = ratios * adv[index]
                surr2 = torch.clamp(ratios, 1 - self.args.epsilon, 1 + self.args.epsilon) * adv[index]
                actor_loss = -torch.min(surr1, surr2) - self.args.entropy_coef * dist_entropy

                self.optimizer_actor.zero_grad()
                actor_loss.mean().backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.optimizer_actor.step()

                v_s = self.critic(s[index])
                critic_loss = F.mse_loss(v_target[index], v_s)

                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer_critic.step()

        # Learning Rate Decay
        if self.args.use_lr_decay:
            self.lr_decay(total_steps)

    def lr_decay(self, total_steps):
        factor = 1 - total_steps / self.args.max_train_steps
        lr_a_now = self.args.lr_a * factor
        lr_c_now = self.args.lr_c * factor
        for p in self.optimizer_actor.param_groups: p['lr'] = lr_a_now
        for p in self.optimizer_critic.param_groups: p['lr'] = lr_c_now
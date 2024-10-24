import random, os, sys, json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from collections import deque
from sklearn import preprocessing
import itertools

# Training params
GAMMA = 0.9
min_hidden = 512
init_w= 11e-3  #3e-3

class Policy(nn.Module):
    def __init__(self,  state_len,  num_actions, num_frames, transformer_depth=4,
                 GRU_hidden=64, GRU_layers=2, attention_heads=4):
        super(Policy, self).__init__()
        self.state_len = state_len
        self.num_actions = num_actions
        self.num_frames = num_frames
        self.GRU_hidden = GRU_hidden
        self.GRU_layers = GRU_layers
        self.attention_heads = attention_heads

        self.gru1 = nn.GRU(self.state_len, self.GRU_hidden, self.GRU_layers)

        self.head = nn.Linear(self.GRU_hidden, self.num_actions)
        self.head.weight.data.uniform_(-init_w, init_w)
        self.head.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, hidden):
        x = state
        x, h = self.gru1(x, hidden)
        x = x[:, -1]
        return F.softmax(self.head(x), dim=-1), h

    def get_action(self, state, hidden, g):

        probs, h = self.forward(state, hidden)

        probs_gen = np.squeeze(probs.detach().numpy()).tolist()

        for i in range (self.num_actions):
            if i == 0 or i == 3 or i == 6 or i == 9:
                probs_gen[i] = probs_gen[i]*(g/4)*4
            else:
                probs_gen[i] = probs_gen[i]*((1-g)/8)
				
        probs_gen = np.exp(probs_gen)/sum(np.exp(probs_gen))  # Softmax

        probs_list = np.squeeze(probs.detach().numpy())
        highest_prob_action = np.random.choice(self.num_actions, p=probs_gen)

        log_prob = torch.log(probs.squeeze(0)[highest_prob_action]) # Natural logarithm

        return highest_prob_action, log_prob, h, probs

    def init_hidden(self):
        weight = next(self.parameters()).data
        hidden = weight.new(self.GRU_layers, self.num_frames, self.GRU_hidden).zero_()
        return hidden

class PG_model:
    def __init__(self, state_size, num_actions, num_frames):
        self.policy = Policy(state_size, num_actions, num_frames)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=0.01) # lr=3e-4
        self.reset()

    def discount_rewards(self, rewards, gamma):
        t_steps = np.arange(len(rewards))
        r = rewards * gamma**t_steps
        r = r[::-1].cumsum()[::-1] / gamma**t_steps
        return r

    def update_policy(self):
		
        discounted_rewards = []

        """
        for t in range(len(self.rewards)):
            Gt = 0 
            pw = 0
            for r in self.rewards[t:]:
                Gt = Gt + GAMMA**pw * r
                pw = pw + 1
            discounted_rewards.append(Gt)

        discounted_rewards = torch.tensor(discounted_rewards)
		"""
		
        discounted_rewards = torch.tensor(self.rewards)

        # Normalisation/standardisation
        ###############################

        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

        policy_gradient = []

        for log_prob, Gt in zip(self.log_probs, discounted_rewards):
            policy_gradient.append(-log_prob * Gt)

        self.optimizer.zero_grad()
        policy_gradient = torch.stack(policy_gradient).sum()
        print("Policy_gradient: ", policy_gradient)
        policy_gradient.backward()
        self.optimizer.step()
        self.reset()
    
    def reset(self):
        self.log_probs = []
        self.rewards = []
        self.policy_hidden = self.policy.init_hidden()

    def get_action(self, state, g):
        hidden = self.policy_hidden
        action, log_prob, hidden, probs_gen = self.policy.get_action(state, hidden, g)
        self.policy_hidden = hidden
        self.log_probs.append(log_prob)
        return action, probs_gen

    def push_replay(self, reward):
        self.rewards.append(reward)

    def save_model(self, dirname, index):
        filename = os.path.join(dirname, "policy_model_" + "%02d"%index + ".pt")
        torch.save({ "policy_state_dict": self.policy.state_dict(),
                     "policy_hidden": self.policy_hidden,
                   }, filename)

    def load_model(self, dirname, index):
        filename = os.path.join(dirname, "policy_model_" + "%02d"%index + ".pt")
        if os.path.exists(filename):
            checkpoint = torch.load(filename)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.policy_hidden = checkpoint["policy_hidden"]
            return True
        return False

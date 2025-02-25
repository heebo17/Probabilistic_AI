import numpy as np
import matplotlib.pyplot as plt

import time

import random
import scipy.signal
from gym.spaces import Box, Discrete
import torch
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.autograd import Variable

#docker build --tag task4 .; docker run --rm -v "$(pwd):/results" task4

def discount_cumsum(x, discount):
    """
    Compute  cumulative sums of vectors.

    Input: [x0, x1, ..., xn]
    Output: [x0 + discount * x1 + discount^2 * x2, x1 + discount * x2, ..., xn]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

def combined_shape(length, shape=None):
    """Helper function that combines two array shapes."""
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(sizes, activation, output_activation=nn.Identity):
    """The basic multilayer perceptron architecture used."""
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


class MLPCategoricalActor(nn.Module):
    """A class for the policy network."""

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        """Takes the observation and outputs a distribution over actions."""
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        """
        Take a distribution and action, then gives the log-probability of the action
        under that distribution.
        """
        return pi.log_prob(act)

    def forward(self, obs, act=None):
        """
        Produce action distributions for given observations, and then compute the
        log-likelihood of given actions under those distributions.
        """
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPCritic(nn.Module):
    """The network used by the value function."""
    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        # Critical to ensure v has right shape
        return torch.squeeze(self.v_net(obs), -1)



class MLPActorCritic(nn.Module):
    """Class to combine policy and value function neural networks."""

    def __init__(self,
                 hidden_sizes=(64,64), activation=nn.Tanh):
        super().__init__()

        obs_dim = 8

        # Build policy for 4-dimensional action space
        self.pi = MLPCategoricalActor(obs_dim, 4, hidden_sizes, activation)

        # Build value function
        self.v  = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, state):
        """
        Take an state and return action, value function, and log-likelihood
        of chosen action.
        """
        # TODO: Implement this function.
        # It is supposed to return three numbers:
        #    1. An action sampled from the policy given a state (0, 1, 2 or 3)
        #    2. The value function at the given state
        #    3. The log-probability of the action under the policy output distribution
        # Hint: This function is only called during inference. You should use
        # `torch.no_grad` to ensure that it does not interfer with the gradient computation.
        
        #SHOULD BE CORRECT
        with torch.no_grad():
            a, _ = self.pi.forward(state)

            number_1 = a.sample()
            number_2 = self.v(state)
            _, number_3 = self.pi.forward(state, number_1)
        return number_1.item(), number_2, number_3
  
    def act(self, state):
        return self.step(state)[0]


class VPGBuffer:
    """
    Buffer to store trajectories.
    """
    def __init__(self, obs_dim, act_dim, size, gamma, lam):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        # calculated TD residuals
        self.tdres_buf = np.zeros(size, dtype=np.float32)
        # rewards
        self.rew_buf = np.zeros(size, dtype=np.float32)
        # trajectory's remaining return
        self.ret_buf = np.zeros(size, dtype=np.float32)
        # values predicted
        self.val_buf = np.zeros(size, dtype=np.float32)
        # log probabilities of chosen actions under behavior policy
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma = gamma
        self.lam = lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append a single timestep to the buffer. This is called at each environment
        update to store the outcome observed outcome.
        """
        # buffer has to have room so you can store
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def end_traj(self, last_val=0):
        """
        Call after a trajectory ends. Last value is value(state) if cut-off at a
        certain state, or 0 if trajectory ended uninterrupted
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # TODO: Implement TD residual calculation.
        # Hint: we do the discounting for you, you just need to compute 'deltas'.
        # see the handout for more info
        
        deltas = rews[:-1] + vals[1:] -vals[:-1]

        #DELTA = reward_t+1 + gamma*V_s_t+1 -V_s
        self.tdres_buf[path_slice] = discount_cumsum(deltas, self.gamma*self.lam)

        #TODO: compute the discounted rewards-to-go. Hint: use the discount_cumsum function
        
        rewards = discount_cumsum(rews[:-1], self.gamma)
        self.ret_buf[path_slice] = rewards
       
        

    def get(self):
        """
        Call after an epoch ends. Resets pointers and returns the buffer contents.
        """
        # Buffer has to be full before you can get something from it.
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0

        # TODO: Normalize the TD-residuals in self.tdres_buf
        self.tdres_buf = self.tdres_buf
        mean = self.tdres_buf.mean()
        std = self.tdres_buf.std()
        self.tdres_buf = (self.tdres_buf - mean)/std

        
        mean_re = self.ret_buf.mean()
        std_re = self.ret_buf.std()
        self.ret_buf = (self.ret_buf - mean_re)/std_re
        

        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    tdres=self.tdres_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}


class Agent:
    def __init__(self, env):
        self.env = env
        self.hid = 64  # layer width of networks
        self.l = 2  # layer number of networks
        # initialises an actor critic
        self.ac = MLPActorCritic(hidden_sizes=[self.hid]*self.l)

    def train(self):
        """
        Main training loop.

        IMPORTANT: This function called by the checker to train your agent.
        You SHOULD NOT change the arguments this function takes and what it outputs!
        """

        # The observations are 8 dimensional vectors, and the actions are numbers,
        # i.e. 0-dimensional vectors (hence act_dim is an empty list).
        obs_dim = [8]
        act_dim = []

        # Training parameters
        # You may wish to change the following settings for the buffer and training
        # Number of training steps per epoch
        steps_per_epoch = 3000
        # Number of epochs to train for
        epochs = 50
        # The longest an episode can go on before cutting it off
        max_ep_len = 300
        # Discount factor for weighting future rewards
        gamma = 0.99
        lam = 0.97

        # Learning rates for policy and value function
        pi_lr = 3e-3
        vf_lr = 1e-3

        # Set up buffer
        buf = VPGBuffer(obs_dim, act_dim, steps_per_epoch, gamma, lam)

        # Initialize the ADAM optimizer using the parameters
        # of the policy and then value networks
        # TODO: Use these optimizers later to update the policy and value networks.
        pi_optimizer = Adam(self.ac.pi.parameters(), lr=pi_lr)
        v_optimizer = Adam(self.ac.v.parameters(), lr=vf_lr)

        # Initialize the environment
        state, ep_ret, ep_len = self.env.reset(), 0, 0

        # Main training loop: collect experience in env and update / log each epoch
        for epoch in range(epochs):
            ep_returns = []
            for t in range(steps_per_epoch):
                a, v, logp = self.ac.step(torch.as_tensor(state, dtype=torch.float32))

                next_state, r, terminal = self.env.transition(a)
                ep_ret += r
                ep_len += 1

                # Log transition
                buf.store(state, a, r, v, logp)

                # Update state (critical!)
                state = next_state

                timeout = ep_len == max_ep_len
                epoch_ended = (t == steps_per_epoch - 1)

                if terminal or timeout or epoch_ended:
                    # if trajectory didn't reach terminal state, bootstrap value target
                    if epoch_ended:
                        _, v, _ = self.ac.step(torch.as_tensor(state, dtype=torch.float32))
                    else:
                        v = 0
                    if timeout or terminal:
                        ep_returns.append(ep_ret)  # only store return when episode ended
                    buf.end_traj(v)
                    state, ep_ret, ep_len = self.env.reset(), 0, 0

            mean_return = np.mean(ep_returns) if len(ep_returns) > 0 else np.nan
            print(f"Epoch: {epoch+1}/{epochs}, mean return {mean_return}")

            # This is the end of an epoch, so here is where you likely want to update
            # the policy and / or value function.
            # TODO: Implement the polcy and value function update. Hint: some of the torch code is
            # done for you.

            data = buf.get()
            #Do 1 policy gradient update
            pi_optimizer.zero_grad() #reset the gradient in the policy optimizer
            obs = data["obs"]
            act = data["act"]
            rew = data["ret"]
            td = data["tdres"]

            #LOSS = REWARD * LOG( PI( a|s ))
            # USE TD AS LOSS
            _, logp = self.ac.pi(obs, act)
            action_loss = -(td*logp).sum()
            action_loss.backward()
            pi_optimizer.step() 
            
            #Hint: you need to compute a 'loss' such that its derivative with respect to the policy
            #parameters is the policy gradient. Then call loss.backwards() and pi_optimizer.step()

            #We suggest to do 100 iterations of value function updates
            rewards = Variable(rew, requires_grad=True)
            
            for _ in range(100):
                v_optimizer.zero_grad()
                values = self.ac.v(obs)
                critic_loss = (values - rewards).pow(2).sum()
                critic_loss.backward()
                v_optimizer.step()
            
            
            
        return True


    def get_action(self, obs):
        """
        Sample an action from your policy.

        IMPORTANT: This function called by the checker to evaluate your agent.
        You SHOULD NOT change the arguments this function takes and what it outputs!
        """
        # TODO: Implement this function.
        # Currently, this just returns a random action.

        #Greedy action selection

        eps=0
        if random.random() > eps:
            #return argmax of Q
            action = self.ac.pi(torch.from_numpy(obs).float())[0].logits.argmax()
            return action
        else: 
            return np.random.choice([0, 1, 2, 3])


def main():
    """
    Train and evaluate agent.

    This function basically does the same as the checker that evaluates your agent.
    You can use it for debugging your agent and visualizing what it does.
    """
    from lunar_lander import LunarLander
    from gym.wrappers.monitoring.video_recorder import VideoRecorder

    env = LunarLander()

    agent = Agent(env)
    agent.train()

    rec = VideoRecorder(env, "policy.mp4")
    episode_length = 300
    n_eval = 10
    returns = []
    print("Evaluating agent...")

    for i in range(n_eval):
        print(f"Testing policy: episode {i+1}/{n_eval}")
        state = env.reset()
        cumulative_return = 0
        # The environment will set terminal to True if an episode is done.
        terminal = False
        env.reset()
        for t in range(episode_length):
            if i <= 10:
                rec.capture_frame()
            # Taking an action in the environment
            action = agent.get_action(state)
            state, reward, terminal = env.transition(action)
            cumulative_return += reward
            if terminal:
                break
        returns.append(cumulative_return)
        print(f"Achieved {cumulative_return:.2f} return.")
        if i == 10:
            rec.close()
            print("Saved video of 10 episodes to 'policy.mp4'.")
    env.close()
    print(f"Average return: {np.mean(returns):.2f}")

if __name__ == "__main__":
    main()

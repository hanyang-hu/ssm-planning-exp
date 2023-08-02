import gymnasium as gym
import torch
import numpy as np
import torch.nn.functional as F
from torch.distributions import Categorical
from dynamics_learning import DynamicsModel
from dataset import FeatureExtraction


class MPC:
    def __init__(self, action_dim, dynamics_model, E, N, L, M, gamma, alpha, device):
        self.initial_dist = Categorical(torch.tensor([0.15, 0.4, 0.15, 0.15, 0.15]).to(device)) 
        self.action_dim = action_dim
        self.model = dynamics_model
        self.E, self.N, self.L, self.M = E, N, L, M
        self.gamma = gamma
        self.alpha = alpha
        self.device = device

    def update(self, dist, sample):
        v = dist.probs
        sample = sample.reshape(-1, self.action_dim)
        freq = sample.sum(axis=0) / sample.sum()
        return Categorical(self.alpha * v + (1 - self.alpha) * freq)

    def take_action(self, initial_state):
        dist = self.initial_dist
        optim_action = 1
        for _ in range(self.E):
            action_sequence_lst = [F.one_hot(dist.sample(torch.Size([self.L])), num_classes = self.action_dim).cpu() for _ in range(self.N)]
            return_lst = []
            for action_sequence in action_sequence_lst:
                state, cum_rew = initial_state, 0.0
                for i in range(self.L):
                    action = action_sequence[i].to(self.device)
                    state, reward = self.model.forward(state, action)
                    cum_rew = cum_rew + reward.item() * (self.gamma ** i)
                return_lst.append(cum_rew)
            idx = np.argsort(np.array(return_lst))[-self.M:]
            sample = torch.zeros(self.M, self.L, self.action_dim).to(self.device)
            for i, id in enumerate(idx):
                sample[i] = action_sequence_lst[id].to(self.device)
            dist = self.update(dist, sample)
            optim_action = torch.argmax(action_sequence_lst[idx[-1]][0], dim = -1).item()

        return optim_action, dist
    

def test_agent(agent, env, n_episode, transform = None, device = None):
    return_list = []
    for episode in range(n_episode):
        print("Episode", episode + 1)
        episode_return = 0
        state, _ = env.reset()
        done = truncated = False
        while not (done or truncated):
            if transform and device:
                state = transform(torch.tensor(state).to(device))
            action, _ = agent.take_action(state)
            print(action)
            next_state, reward, done, truncated, _ = env.step(action)
            state = next_state
            episode_return += reward
            env.render()
        return_list.append(episode_return)
    return np.mean(return_list)


if __name__ == '__main__':

    env = gym.make('highway-v0', render_mode = 'rgb_array')
    config = {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 40,
            "features": ["presence", "x", "y", "vx", "vy"],
            "features_range": {
                "x": [-100, 100],
                "y": [-100, 100],
                "vx": [-20, 20],
                "vy": [-20, 20]
            },
            "absolute": False
        }
    }
    env.configure(config)
    torch.manual_seed(0)
    np.random.seed(0)

    state_dim = 128
    action_dim = 5
    latent_dim = 196
    hidden_dim = [256, 256]
    lr = 1e-3
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_name = "ssm_sample.pt"

    dynamics_model = DynamicsModel(state_dim, action_dim, latent_dim, hidden_dim, device)
    dynamics_model.load_state_dict(torch.load("./models/" + model_name))
    dynamics_model.eval()

    E, N, L, M = 2, 3, 4, 1 # Update CEM for E times, each time sample N action sequences of length L, choose top M
    gamma = 0.8
    alpha = 0.0 # soft update of categorical distribution

    mpc_agent = MPC(action_dim, dynamics_model, E, N, L, M, gamma, alpha, device)
    transform = FeatureExtraction().attn.to(device)

    print("Average return: {}".format(test_agent(mpc_agent, env, 50, transform, device)))
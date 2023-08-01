import torch
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torch.distributions.kl import kl_divergence
import torch.optim.lr_scheduler as lr_scheduler

class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        layer_dim = [input_dim,] + hidden_dim + [output_dim,]
        self.fc = torch.nn.ParameterList([torch.nn.Linear(layer_dim[i], layer_dim[i+1]) for i in range(len(layer_dim) - 1)])

        for layer in self.fc:
            torch.nn.init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0)
        
    def forward(self, x):
        for layer in self.fc:
            x = F.tanh(layer(x))
        return x
    

class PNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PNN, self).__init__()
        self.mlp = MLP(input_dim, hidden_dim, output_dim * 2)

    def forward(self, x):
        mu, log_var = self.mlp(x).chunk(2, dim=-1)
        std = torch.exp(0.5 * log_var)
        return MultivariateNormal(mu, torch.diag_embed(std))


class DynamicsModel(torch.nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim, hidden_dim, device, N = 1000):
        super(DynamicsModel, self).__init__()

        self.q = PNN(state_dim, hidden_dim, latent_dim).to(device)
        self.p_t = PNN(latent_dim + action_dim, hidden_dim, latent_dim).to(device)
        self.p_o = PNN(latent_dim, hidden_dim, state_dim).to(device)
        self.p_r = PNN(latent_dim, hidden_dim, 1).to(device)

        self.device = device
        self.N = N # number of runs of using Monte Carlo to calculate expectation

    def forward(self, s, a):
        z = self.q(s).rsample() 
        n_z = self.p_t(torch.cat((z, a), dim=-1)).rsample() 
        n_s = self.p_o(n_z).rsample() 
        n_r = self.p_r(n_z).rsample() 
        return n_s, n_r # return predicted state and reward

    def loss(self, transition_dict):
        states = transition_dict['state'].to(self.device) # s_{t-1}
        actions = transition_dict['action'].to(self.device) # a_{t-1}
        rewards = transition_dict['reward'].view(-1, 1).to(self.device) # r_t
        next_states = transition_dict['next_state'].to(self.device) # s_t

        latents_dist = self.q(states)
        latents = latents_dist.rsample() # z_{t-1}
        next_latents_dist = self.q(next_states)
        next_latents = next_latents_dist.rsample() # z_t

        rec_loss = self.p_o(next_latents).log_prob(states).mean() + self.p_r(next_latents).log_prob(rewards).mean()

        kl_loss = kl_divergence(next_latents_dist, self.p_t(torch.cat((latents, actions), dim=-1))).mean()

        return -rec_loss + kl_loss


if __name__ == '__main__':
    
    from dataset import TransformDataset
    from torch.utils.data import DataLoader
    from matplotlib import pyplot as plt
    import numpy as np
    from tqdm import tqdm

    state_dim = 128
    action_dim = 5
    latent_dim = 128
    hidden_dim = [256, 256]
    lr = 1e-3
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = DynamicsModel(state_dim, action_dim, latent_dim, hidden_dim, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    # scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=100)

    batch_size = 128
    n_epochs = 100
    n_iterations = 10

    dataset = TransformDataset('transition_data.pkl')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=16)
    iterator = list(iter(dataloader))

    total_loss_lst = []

    for epoch in range(n_epochs):
        with tqdm(total=n_iterations, desc="Epoch %i: " % (epoch + 1)) as pbar:
            for _ in range(n_iterations):
                loss_lst = []
                for batch in iterator:
                    optimizer.zero_grad()
                    loss = model.loss(batch)
                    loss.backward()
                    optimizer.step()
                    loss_lst.append(loss.detach().item())
                # scheduler.step()
                total_loss_lst = total_loss_lst + loss_lst
                pbar.set_postfix({'loss': '%.3f' % (np.mean(loss_lst))})
                pbar.update(1)

    torch.save(model.state_dict(), "./models/ssm.pt")

    def moving_average(a, window_size):
        cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
        middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
        r = np.arange(1, window_size-1, 2)
        begin = np.cumsum(a[:window_size-1])[::2] / r
        end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
        return np.concatenate((begin, middle, end))

    mv_return = moving_average(total_loss_lst, 9)
    plt.plot(list(range(len(total_loss_lst))), mv_return)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('ELBO Loss for Dynamics Learning')
    plt.savefig('./figures/moving_avg_loss.png')

    print("Training completed!!!")

    
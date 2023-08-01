import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import math
import pickle
from dynamics_learning import MLP

class Attention(torch.nn.Module):
    def __init__(self, ego_dim, oppo_dim, embed_dim = 128, num_heads = 4):
        super(Attention, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = MLP(ego_dim, [128,], embed_dim)
        self.kv_proj = MLP(oppo_dim, [128,], 2*embed_dim)
        self.o_proj = MLP(embed_dim, [], embed_dim) # Only 1 linear layer
        
    def forward(self, x):
        if len(x.shape) > 2:
            ego, oppo = x[:,0,:], x[:,:,:]
        else:
            ego, oppo = x[0], x[:]

        # Accept both batched and unbatched input
        is_batched = ego.dim() > 1
        if not is_batched:
            ego = ego.unsqueeze(0)
            oppo = oppo.unsqueeze(0)
        batch_size = ego.size(0)

        # Compute and separate Q, K, V from linear output
        q = self.q_proj(ego).reshape(batch_size, self.num_heads, 1, self.head_dim)
        k, v = self.kv_proj(oppo).reshape(batch_size, oppo.size()[1], self.num_heads, 2*self.head_dim).permute(0, 2, 1, 3).chunk(2, dim=-1)

        # Determine value and attention outputs
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) # broadcasting
        attn_logits = attn_logits / math.sqrt(self.head_dim) # d_k == head_dim
        attention = F.softmax(attn_logits, dim=-1)
        values = torch.matmul(attention, v).reshape(batch_size, self.embed_dim)
        o = self.o_proj(values)

        return o if is_batched else o.squeeze(0)
    

class FeatureExtraction(object):
    def __init__(self, attn_model_path = "./models/attn_feature.pt"):
        self.attn = Attention(5, 5)
        try:
            model = torch.load(attn_model_path)
            self.attn.load_state_dict(model)
            for param in self.attn.parameters():
                param.requires_grad = False
        except:
            raise

    def __call__(self, sample):
        s, a, r, n_s, d  = sample['state'], sample['action'], sample['reward'], sample['next_state'], sample['done']
        return {'state': self.attn(torch.tensor(s)), 
                'action': F.one_hot(torch.tensor(a), num_classes = 5), 
                'reward': torch.tensor(r),
                'next_state': self.attn(torch.tensor(n_s)),
                'done': torch.tensor(d)}


class TransformDataset(Dataset):
    def __init__(self, pkl_file, root_dir = "./data/"):
        self.trajectory_data = pickle.load(open(root_dir + pkl_file, 'rb'))
        self.transform = FeatureExtraction()

    def __len__(self):
        return len(self.trajectory_data)
    
    def __getitem__(self, idx):
        return self.transform(self.trajectory_data[idx])
import torch
import torch.nn as  nn
from torch.nn import functional as F

with open('data/input.txt', 'r', encoding='utf-8') as f:
  text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

#encoder and decoder for characters
# can be replaced by a tokenizer like TikToken
encode  = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])


# encode input data as torch tensors

data = torch.tensor(encode(text), dtype = torch.long)

#split data into train and validation pairs

n  = int(0.9*len(data))
train_data = data[:n]
test_data = data[n:]

# create mini batches for multiple chunks of text that are stacked up in a single tensor
# for parallel processing of data

torch.manual_seed(1337)
batch_size = 4 # how many sequences processed in parallel
block_size = 8 # maximum context length for predictions

def get_batch(split):
  data= train_data if split == 'train' else test_data
  ix = torch.randint(len(data) - block_size, (batch_size,))
  # take 1d tensors as a row
  x = torch.stack([data[i:i+block_size] for i in ix])
  y = torch.stack([data[i+1:i+block_size+1] for i in ix])
  return x,y




###### Hyperparameters ########
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 300
learning_rate = 3e-4

eval_iters = 200
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embed = 384
n_head = 6
n_layer =6
dropout = 0.2
###### Hyperparameters ########


@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out



class Head(nn.Module):

  def __init__(self, head_size):
    super().__init__()

    self.key = nn.Linear(n_embed, head_size, bias = False)
    self.query = nn.Linear(n_embed, head_size, bias = False)
    self.value = nn.Linear(n_embed, head_size, bias = False)
    self.register_buffer('tril', torch.tril(torch.ones(block_size,block_size)))

    self.dropout = nn.Dropout(dropout)

  def forward(self,x):
    B,T,C = x.shape
    k = self.key(x) # (B,T,C)
    q = self.query(x) # (B,T,C)

    # perform scaled attention
    wei = q @ k.transpose(-2,-1) * C**(-0.5) # (B,T,C) @ (B,C,T) -> (B,T,T)
    wei = wei.masked_fill(self.tril[:T,:T] ==0, float('-inf')) # (B,T,T)
    wei = F.softmax(wei,dim=-1)

    wei = self.dropout(wei)

    v = self.value(x)
    out = wei @ v
    return out


class MultiHeadAttention(nn.Module):

  def __init__ (self, num_heads, head_size):
    super().__init__()

    self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    self.proj = nn.Linear(num_heads * head_size, n_embed)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    out =  torch.cat([h(x) for h in self.heads], dim=-1)
    out =  self.dropout(self.proj(out))
    return out


class FeedForward(nn.Module):
  """ a simple linear layer followed by a non-linearity """

  def __init__(self, n_embed):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(n_embed, 4 * n_embed),
        nn.ReLU(), # for residual connections i guess
        nn.Linear(4 * n_embed, n_embed),
        nn.Dropout(dropout),
    )

  def forward(self, x):
    return self.net(x)


class Block(nn.Module):
  """ Transformer block: communication followed by computation """

  def __init__(self, n_embed, n_head):

    super().__init__()
    head_size = n_embed // n_head

    self.sa = MultiHeadAttention(n_head, head_size)
    self.ffwd = FeedForward(n_embed)
    # makes it unit gaussian at initiation
    self.ln1 = nn.LayerNorm(n_embed)
    self.ln2 = nn.LayerNorm(n_embed)

  def forward(self, x):
    # residual connections
    x = x + self.sa(self.ln1(x))
    x= x + self.ffwd(self.ln2(x))
    return x


class BigramLM(nn.Module):

  def __init__(self):
    super().__init__()

    self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
    self.position_embedding_table = nn.Embedding(block_size, n_embed)
    self.blocks = nn.Sequential(*[Block(n_embed, n_head = n_head) for _ in range(n_layer)])
    self.ln_f = nn.LayerNorm(n_embed)
    self.lm_head = nn.Linear(n_embed, vocab_size)

  def forward(self, idx, targets = None):
    B,T = idx.shape
    idx = idx.cuda() if torch.cuda.is_available() else idx
    tok_emb = self.token_embedding_table(idx)
    pos_emb = self.position_embedding_table(torch.arange(T,device = device))
    x = tok_emb + pos_emb
    x= self.blocks(x)
    x = self.ln_f(x)
    logits = self.lm_head(x)

    if targets is None:
      loss = None
    else:
      B,T,C = logits.shape
      logits = logits.view(B*T, C)
      logits = logits.cuda() if torch.cuda.is_available() else logits
      targets = targets.view(B*T)
      
      targets = targets.cuda() if torch.cuda.is_available() else targets
      loss = F.cross_entropy(logits, targets)

    return logits, loss

  def generate(self, idx, max_new_tokens):

    for _ in range(max_new_tokens):
      # crop idx so that positional embedding doesnt run out of scope
      idx_cond = idx[:, -block_size:]
      # get predictions
      logits, loss = self(idx_cond)
      # pick the last time step
      logits = logits[:,-1,:]
      # apply softmax to get probabilities
      probs = F.softmax(logits,dim=-1)
      # sample from the distribution (pick the best)
      idx_next = torch.multinomial(probs, num_samples=1)
      # GPT like output
      print(decode(idx_next[0].tolist()), end='')
      # append sampled index to running sequence
      idx = torch.cat((idx, idx_next), dim=1)
      
    return idx

def train():

  model = BigramLM()
  m = model.to(device)
  # create a PyTorch optimizer
  optimizer = torch.optim.AdamW(model.parameters(),lr=1e-3)

  for iter in range(max_iters):

    if iter % eval_interval == 0:
      losses = estimate_loss(model)
      print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = model(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    torch.save(model, 'saved_model.pth')

if __name__ == "__main__":
  train()


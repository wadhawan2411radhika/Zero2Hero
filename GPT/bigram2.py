import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size = 32
block_size = 8
max_iters = 5000
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
eval_interval = 300
n_embed = 384
n_head = 6
n_layer = 6
dropout = 0.2
torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

# Tokenize the text, mapping from characters to integers
stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] # string -> list of integers
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

T = 8

def get_batch(split):
    data = train_data if split=='train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # all of these could be potential target over some context
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
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
        self.key = nn.Linear(n_embed, head_size, bias=False) 
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # not a trainable parameter
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T]==0, float('-inf')) # Self attention weights with masked causal LM
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out
    
# Self-attention heads help updating the embeddings as per context
# but we havent really interacted with updated vectors, each token individually within itself
# Point wise feed forward NN
class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = self.ln1(x)
        x = x + self.sa(x) # Residual connections
        x = self.ln2(x)
        x = x + self.ffwd(x)
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # nn.Embedding is nothing but lookup table for given indices, intialised randomly, learnable parameters(weights/ embddings)
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed) # thin wrapper on tensor
        self.position_embedding_table = nn.Embedding(block_size, n_embed) # block_size = context window
        # self.sa_heads = MultiHeadAttention(4, n_embed//4) # similar to applying multiple convolution filters -> each extracting something unique
        # self.ffwd = FeedForward(n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)]) 
        # Become very Deep NN, need resurrection -> Residual Connection + Layer Norm
        
        # Adding a linear layer here - FC Layer -> final output = size of(vocab)
        # This creates a one stage of interaction of an embedding with surroudnding indices
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B, T) tensor of integers
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B, T, C) - batch=4 x time=8 x channel=vocab size
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # will lookup randokmly initialised rows for each position
        x = tok_emb + pos_emb # (B, T, C) + (T, C) -> broadcasting will work
        # x = self.sa_heads(x)
        # x = self.ffwd(x)
        x = self.blocks(x)

        logits = self.lm_head(x) # (B, T, vocab_size), logits = z = post one FC layer, visualise this as a bert each token has output of embed_dim 
        if targets is None:
            loss = None
        else:
            # Cross Entropy expects (B, C, T)
            B, T, C = logits.shape
            logits = logits.view(B*T, C) # becomes 2 Dimensional
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:] # only take recent t elements
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] # becomes B, C
            
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)

            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) #(B, T+1)
        return idx # appended token ids
    
model = BigramLanguageModel()
m = model.to(device)

optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)
for iter in range(max_iters):

    if iter % eval_interval ==0:
        losses = estimate_loss()
        print(f"Step: {iter}, Train Loss: {losses['train']}, Val Loss: {losses['val']}")
    # get the data
    xb, yb = get_batch('data')

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step() # Updating the parameters
    # losses.append(loss.item())

# Generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=1000)[0].tolist()))
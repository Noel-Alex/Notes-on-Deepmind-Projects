
28-10-2024 15:40

Status:

Tags: [[Ai]] [[LLM-LLVM]] 


# What are GPTs

The idea of [transformers](Terms#Transformers) were first introduced in Google's groundbreaking paper titled [Attention is all you need](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) which has since been cited over 138,000 times and it outlines the approach on how you could train ai models with massive parameter counts in parallel saving huge amounts of time as they could be computed using GPUs.![[Pasted image 20241028155139.png]]

A very rudimentary implementation of this architecture is given below, but if you want to build it yourself I strongly recommend you to follow along with [Andrej Karpathy](https://youtu.be/kCc8FmEb1nY?si=jJIS7j6SY-8aeNnU) or this tutorial from [freecodecamp](https://www.youtube.com/watch?v=UU1WVnMk4E8&pp=ygUQZ3B0IGZyb20gc2NyYXRjaA%3D%3D) 

```python
import mmap  
import random  
import time  
  
import torch  
import torch.nn as nn  
from torch.nn import functional as F  
import pickle  
from torch.cuda.amp import autocast, GradScaler  
  
#Hyperparameters  
device = 'cuda' if torch.cuda.is_available() else 'cpu'  
block_size = 128  
batch_size = 48  
max_iters = 100000  
learning_rate = 3e-5  
eval_iters = 100  
n_embd = 384  
n_head = 20  
n_layer = 16  
dropout = 0.2  
scaler = GradScaler()  
  
  
  
  
chars = ''  
with open('vocab.txt', 'r', encoding = 'utf-8') as f:  
    text = f.read()  
    chars = sorted(list(set(text)))  
vocab_size = len(chars)  
  
string_to_int = {ch: i for i, ch in enumerate(chars)}  
int_to_string = {i: ch for i, ch in enumerate(chars)}  
encode = lambda s: [string_to_int[c] for c in s]  
decode = lambda k: ''.join([int_to_string[i] for i in k])  
data = torch.tensor(encode(text), dtype=torch.long)  
  
  
def get_random_chunk(split):  
    filename = 'train_split.txt' if split == 'train' else ('val_split.txt')  
    with open(filename, 'rb') as f:  
        with mmap.mmap(f.fileno(), 0, access = mmap.ACCESS_READ) as mm:  
            file_size = len(mm)  
            start_pos = random.randint(0, (file_size)- block_size*batch_size)  
  
            mm.seek(start_pos)  
            block = mm.read(block_size*batch_size-1)  
  
            decoded_block = block.decode('utf-8', errors = 'ignore').replace('\r', '')  
  
            data = torch.tensor(encode(decoded_block), dtype=torch.long)  
    return data  
def get_batch(split):  
    data = get_random_chunk(split)  
    ix = torch.randint(len(data) - block_size, (batch_size,))  
    x = torch.stack([data[i:i+block_size]for i in ix])  
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])  
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
  
  
class FeedForward(nn.Module):  
    def __init__(self, n_embd):  
  
        super().__init__()  
        self.net = nn.Sequential(  
            nn.Linear(n_embd, 4 * n_embd),  
            nn.ReLU(),  
            nn.Linear(4 * n_embd, n_embd),  
            nn.Dropout(dropout))  
  
    def forward(self, x):  
        return self.net(x)  
  
  
class Head(nn.Module):  
    """ one head of self-attention """  
  
    def __init__(self, head_size):  
        super().__init__()  
        self.key = nn.Linear(n_embd, head_size, bias=False)  
        self.query = nn.Linear(n_embd, head_size, bias=False)  
        self.value = nn.Linear(n_embd, head_size, bias=False)  
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))  
  
        self.dropout = nn.Dropout(dropout)  
  
    def forward(self, x):  
        # input of size (batch, time-step, channels)  
        # output of size (batch, time-step, head size)        B,T,C = x.shape  
        k = self.key(x)   # (B,T,hs)  
        q = self.query(x) # (B,T,hs)  
        # compute attention scores ("affinities")        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)  
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)  
        wei = F.softmax(wei, dim=-1) # (B, T, T)  
        wei = self.dropout(wei)  
        # perform the weighted aggregation of the values  
        v = self.value(x) # (B,T,hs)  
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)  
        return out  
  
  
class MultiHeadAttention(nn.Module):  
    """ multiple heads of self-attention in parallel """  
  
    def __init__(self, num_heads, head_size):  
        super().__init__()  
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])  
        self.proj = nn.Linear(head_size * num_heads, n_embd)  
        self.dropout = nn.Dropout(dropout)  
  
    def forward(self, x):  
        out = torch.cat([h(x) for h in self.heads], dim=-1) # (B, T, F) -> (B, T, [h1, h1, h1, h1, h2, h2, h2, h2, h3, h3, h3, h3])  
        out = self.dropout(self.proj(out))  
        return out  
  
class Block(nn.Module):  
    def __init__(self, n_embd, n_head):  
        #n_embd embedding dimensions, n_head:the number of heads we'd like  
        super().__init__()  
        head_size = n_embd // n_head  
        self.sa = MultiHeadAttention(n_head, head_size)  
        self.ffwd = FeedForward(n_embd)  
        self.ln1 = nn.LayerNorm(n_embd)  
        self.ln2 = nn.LayerNorm(n_embd)  
    def forward(self, x):  
        y = self.sa(x)  
        x = self.ln1(x+y)  
        y = self.ffwd(x)  
        x = self.ln2(x+y)  
        return x  
class GPTLanguageModel(nn.Module):  
    def __init__(self, vocab_size):  
        super().__init__()  
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)  
        self.position_embedding_table = nn.Embedding(block_size, n_embd)  
        self.blocks = nn.Sequential(*[Block(n_embd, n_head = n_head) for _ in range(n_layer)])  
  
        self.ln_f = nn.LayerNorm(n_embd) #final layer Norm  
        self.lm_head = nn.Linear(n_embd, vocab_size)  
  
        self.apply(self._init_weights)  
  
    def _init_weights(self, module):  
        if isinstance(module, nn.Linear):  
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)  
            if module.bias is not None:  
                torch.nn.init.zeros_(module.bias)  
        elif isinstance(module, nn.Embedding):  
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)  
  
    def forward(self, index, targets=None):  
        B, T = index.shape  
        #index and targets are both (B, T) tensors  
        tok_emb = self.token_embedding_table(index)  #(B,T,C)  
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) #T,C  
        x = tok_emb + pos_emb #(B, T, C)  
        x = self.blocks(x)#(B, T, C)  
        x = self.ln_f(x)#(B, T, C)  
        logits = self.lm_head(x)#(B, T, vocab_size)  
        if targets is None:  
            loss = None  
        else:  
            B, T, C = logits.shape  
            logits = logits.view(B*T, C)  
            targets = targets.view(B*T)  
            loss = F.cross_entropy(logits, targets)  
  
        return logits, loss  
  
    def generate(self, index, max_new_tokens):  
        # index is (B, T) array of indices in the current context  
        for _ in range(max_new_tokens):  
            # crop idx to the last block_size tokens  
            index_cond = index[:, -block_size:]  
            # get the predictions  
            logits, loss = self.forward(index_cond)  
            # focus only on the last time step  
            logits = logits[:, -1, :]  # becomes (B, C)  
            # apply softmax to get probabilities            probs = F.softmax(logits, dim=-1)  # (B, C)  
            # sample from the distribution            index_next = torch.multinomial(probs, num_samples=1)  # (B, 1)  
            # append sampled index to the running sequence            index = torch.cat((index, index_next), dim=1)  # (B, T+1)  
        return index  
  
  
x, y = get_batch('train')  
model = GPTLanguageModel(vocab_size)  
  
print('Loading model')  
with open('model-01.pkl', 'rb') as f:  
    model = pickle.load(f)  
print('Model loaded')  
m = model.to(device)  
  
#print('generating text')  
'''context = torch.zeros((1,1), dtype=torch.long, device=device)  
generated_chars = decode(m.generate(context, max_new_tokens=1000)[0].tolist())  
print(generated_chars)'''  
  
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)  
start = time.perf_counter()  
for iter in range(max_iters):  
    if iter % eval_iters == 0:  
        losses = estimate_loss()  
        end = time.perf_counter()  
        print(f"step:{iter}, train loss: {losses['train']:.4f}, val loss: {losses['val']:.4f} in {(end - start):.4f} seconds")  
        start = time.perf_counter()  
        with open('model-01.pkl', 'wb') as f:  
            pickle.dump(model, f)  
  
        print('model-saved')  
  
    xb, yb = get_batch('train')  
    with autocast():  
        logits, loss = model(xb, yb)  
    optimizer.zero_grad(set_to_none=True)  
    scaler.scale(loss).backward()  
    scaler.step(optimizer)  
    scaler.update()  
print(loss.item())  
  
with open('model-01.pkl', 'wb') as f:  
    pickle.dump(model, f)  
  
print('model-saved')  
'''context = torch.zeros((1,1), dtype=torch.long, device=device)  
generated_chars = decode(m.generate(context, max_new_tokens=1000)[0].tolist())  
print(generated_chars)'''
```


**GPTs (Generative Pre-trained Transformers)** are a family of language models developed by OpenAI, known for their impressive natural language processing abilities. They utilize the Transformer architecture, introduced by Vaswani et al. in 2017, which uses self-attention mechanisms to understand relationships within sequences, making it particularly effective for language understanding and generation.

### Key Features and Evolution

1. **GPT-1**:
    
    - Introduced in 2018, GPT-1 was the first in the series and demonstrated that a transformer-based model could be pre-trained on a large corpus of text and then fine-tuned for specific tasks.
    - With 117 million parameters, it was trained using unsupervised learning on vast internet data, achieving basic language comprehension and generation capabilities.
2. **GPT-2**:
    
    - GPT-2, released in 2019, represented a significant leap with 1.5 billion parameters, producing coherent and contextually appropriate text on diverse topics.
    - It gained attention for generating text that appeared remarkably human, raising concerns about potential misuse, which initially led OpenAI to limit its release.
    - GPT-2 is proficient in tasks like summarization, translation, and even simple reasoning.
3. **GPT-3**:
    
    - Released in 2020, GPT-3 has 175 billion parameters and can perform complex tasks without task-specific training, simply by understanding instructions in natural language (known as "few-shot" or "zero-shot" learning).
    - Its scale and training diversity allowed it to excel in language-based applications like code generation, creative writing, and conversational AI.
4. **GPT-4**:
    
    - Launched in 2023, GPT-4 further improved context handling, fine-tuning abilities, and language translation, among other tasks.
    - OpenAI introduced multimodal capabilities in GPT-4, allowing it to handle both text and images, opening new applications in fields such as visual storytelling, diagnostics, and interactive user interfaces.

### Technical Mechanisms

- **Transformer Architecture**: GPT models use transformers, which rely on self-attention to understand context and relationships within data. This approach enables efficient processing of long text sequences compared to older models like RNNs and LSTMs.
- **Pre-training and Fine-tuning**: GPT models are first pre-trained on extensive corpora to understand general language patterns and then can be fine-tuned or prompted for specific tasks, making them highly versatile.
- **Few-shot Learning**: GPT-3 and later models can perform tasks without extensive training examples, demonstrating the potential for broader applications without task-specific datasets.

### Applications

GPT models have been applied in numerous domains:

- **Conversational Agents**: Chatbots and virtual assistants leverage GPT’s language understanding for natural interactions.
- **Content Generation**: Used for generating articles, summaries, and other creative content, GPTs excel at creating coherent and contextually relevant text.
- **Coding**: GPT models like Codex (derived from GPT-3) specialize in code generation, helping developers by generating or debugging code.
- **Education and Research**: GPT-4’s multimodal capabilities support education, research, and accessibility by providing explanations, generating visual interpretations, and assisting with learning material.

### Ethical Considerations

GPT models raise ethical considerations, such as:

- **Misinformation**: Their ability to generate convincing text can lead to the spread of misinformation if misused.
- **Bias**: Being trained on vast internet data, GPTs can inadvertently reflect biases present in that data.
- **Employment Impact**: The automation of text-based tasks can affect industries that rely on manual content creation and curation.
# References
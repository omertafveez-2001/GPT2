import tiktoken
from gpt2 import *
from modules import *
import torch
from torch.nn import functional as F


device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backwards, "mps") and torch.backwards.mps.is_available():
    device="mps"
print(f"using device: {device}")


num_return_sequences = 5
max_length = 30

# pretrained model initialization
model = GPT.from_pretrained("gpt2")

# Random Torch model intialization
model = GPT(GPTConfig())

model.eval()
model.to(device)

torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Tokenizing the input. 
enc = tiktoken.get_encoding("gpt2") # loading gpt-2 tokenizer
tokens = enc.encode("Hello, I'm a Language model,") # encoding the input
tokens = torch.tensor(tokens, dtype=torch.long) # (8,) # converting the tokens into tensor array
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (5,8) # repeating the tokens 5 times.
x = tokens.to(device) 
print(tokens)


while x.size(1) < max_length:
# forward the model the get the logits
    with torch.no_grad():
        logits = model(x) # (B,T, vocab_size)
        # take the logits at the last position
        logits = logits[:, -1, :] # (B, vocab_size)
        # get the probabilities
        probs = F.softmax(logits, dim=-1)
        # do top-k sampling of 50 (hugging face default)
        # topk_probs here becomes (5,50) topk_indices is (5,50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)

        # select a token from the top-k probabilities
        ix = torch.multinomial(topk_probs, 1) # (8,1)

        # gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix) # (8,1)
        # append to the sequence
        x = torch.cat((x, xcol), dim=1)



for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)
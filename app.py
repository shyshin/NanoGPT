
import torch
from train import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.load('saved_model.pth', map_location= torch.device(device))

num_of_tokens = input("Num of tokens to generate? ")
num_of_tokens = int(num_of_tokens)
temperature = input("Temperature? ")
temperature = float(temperature)
idx = torch.zeros((1,1), dtype=torch.long)
output = decode(model.generate(idx,max_new_tokens=num_of_tokens, temperature=temperature)[0].tolist())


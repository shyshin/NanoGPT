
import torch
from train import BigramLM, encode, decode

with open('data/input.txt', 'r', encoding='utf-8') as f:
  text = f.read()


model = torch.load('saved_model.pth')

num_of_tokens = input("Num of tokens to generate? ")
num_of_tokens = int(num_of_tokens)
idx = torch.zeros((1,1), dtype=torch.long)
print(decode(model.generate(idx,max_new_tokens=num_of_tokens)[0].tolist()))


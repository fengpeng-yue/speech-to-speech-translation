#! /usr/bin/env python3
import torch
import sys

model = torch.load(sys.argv[1])
model["cfg"]["model"].decoder_embed_dim = 512
torch.save(model, sys.argv[2])
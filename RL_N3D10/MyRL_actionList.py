import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import math
import numpy as np
import torch 
import random
from torch import nn
from torch.nn  import functional as F 
from torch.autograd import Variable
from torch.optim import Adam
from torch.distributions.categorical import Categorical

from MyRL_ID import wiresIDSet_brickwall_fun
from MyRL_ID import actionIDSet_fun


from MyRL_attention import CustomTransformerClassifier


def actionlistBW_fun(  Nq, Depth):
	wiresIDSet_BrickWall = wiresIDSet_brickwall_fun(Nq)
	Depth_BW = len(wiresIDSet_BrickWall)
	while Depth_BW < Depth:
		wiresIDSet_BrickWall = wiresIDSet_BrickWall + wiresIDSet_brickwall_fun(Nq)
		Depth_BW = len(wiresIDSet_BrickWall)
	actionList_BrickWall = actionIDSet_fun(wiresIDSet_BrickWall, Nq)
	return actionList_BrickWall[:Depth] 

def actionlistAG_fun(Depth, batch_size, agent):
	action_batch = [[0]]*batch_size
	# agent.eval()
	for D in range(1,Depth):
		logits = agent(action_batch)
		actionNext_batch = Categorical(logits=logits).sample() 
		action_batch = torch.cat( (torch.tensor(action_batch), actionNext_batch.reshape(batch_size, 1)), dim=1).tolist()
	return action_batch


if __name__ == '__main__':
	torch.set_default_dtype(torch.float64)

	num_qubit = 4
	Depth = 6	
	actionlistBW = actionlistBW_fun(  Nq=4, Depth=6)
	print("\nactionlistBW", actionlistBW)

	Nq = 3
	Depth = 3
	embed_dim = 10
	num_heads = 1
	TransformerClassifier = CustomTransformerClassifier(vocab_size = Nq*Nq+1, max_seq_len=Depth, embed_dim= embed_dim, num_heads=num_heads, ffn_hidden_dim=embed_dim*5, num_classes= Nq*Nq)
	actionlistAG_batch= actionlistAG_fun( Depth=Depth, batch_size=2, agent=TransformerClassifier)
	print("\nactionlistAG_batch", actionlistAG_batch)


	
























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

from MyRL_attention import CustomTransformerClassifier

from MyRL_actionList import actionlistAG_fun
from MyRL_DataSet import DataSet_ini_fun, DataSet_app_fun

from MyFisher_circuit import   measurement_circuit_fun
from MyFisher_Rank import effective_rank_class

def compute_loss_fun(DataXSet, DataYSet, DataRSet, agent):
	batch_size = len(DataXSet)
	logits = agent(DataXSet)
	output = torch.tensor(DataYSet).reshape(batch_size)
	logp = Categorical(logits=logits).log_prob( output	)	
	weight = torch.tensor(DataRSet).reshape(batch_size)
	loss = -(logp * weight)
	return loss.mean()


def train_agent_fun(DataXSet, DataYSet, DataRSet, agent):
	optimizer = Adam(agent.parameters(), lr=0.01)
	epoch_size = 101
	for epo in range(epoch_size) :	
		optimizer.zero_grad()
		batch_loss = compute_loss_fun(DataXSet=DataXSet, DataYSet= DataYSet, DataRSet= DataRSet, agent= agent)
		batch_loss.backward()
		optimizer.step()
		if epo % 20 ==0 or epo == epoch_size-1:
			print("\tepoch=", epo, "/"+str(epoch_size-1)+"\tloss=", batch_loss.item())
	return batch_loss 




if __name__ == '__main__':
	torch.set_default_dtype(torch.float64)

	

	
	Nq = 3
	Depth = 3
	embed_dim = 10
	num_heads = 1
	TransformerClassifier = CustomTransformerClassifier(vocab_size = Nq*Nq+1, max_seq_len=Depth, embed_dim= embed_dim, num_heads=num_heads, ffn_hidden_dim=embed_dim*5, num_classes= Nq*Nq)


	Nq  = Nq
	InputX = torch.tensor(np.load("./Data/rhoXNq"+str(Nq)+"batch"+str(20)+".npy"))
	measurement_circuit_fun = measurement_circuit_fun
	effective_rank_F = effective_rank_class(Nq, InputX, measurement_circuit_fun)

	
	actionlistAG_batch= actionlistAG_fun( Depth=Depth, batch_size=1, agent=TransformerClassifier)
	print("\nactionlistAG_batch", actionlistAG_batch)
	DataXSet, DataYSet, DataRSet,DataXYSet = DataSet_ini_fun(Depth=Depth, actionlist=actionlistAG_batch[0], effective_rank_F=effective_rank_F)
	actionlistAG_batch= actionlistAG_fun( Depth=Depth, batch_size=10, agent=TransformerClassifier)
	# print('\n',DataXYSet,'\n', DataRSet)
	# DataXSet, DataYSet, DataRSet,DataXYSet = DataSet_app_fun(DataXSet= DataXSet,DataYSet= DataYSet,DataRSet=DataRSet,DataXYSet= DataXYSet,  
	# 														Depth=Depth, actionlist_batch=actionlistAG_batch, effective_rank_F=effective_rank_F)
	# print('\n',DataXYSet,'\n', DataRSet)

	batch_loss = compute_loss_fun(DataXSet=DataXSet, DataYSet= DataYSet, DataRSet= DataRSet, agent= TransformerClassifier)
	
	# train_agent_fun(DataXSet=DataXSet, DataYSet=DataYSet, DataRSet=DataRSet, agent=TransformerClassifier)
	



	
























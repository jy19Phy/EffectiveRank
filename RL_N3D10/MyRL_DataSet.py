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

from MyRL_actionList import actionlistBW_fun,actionlistAG_fun

from MyRL_attention import CustomTransformerClassifier

def DataSet_ini_fun(Depth, actionlist, effective_rank_F):
	Depth = len(actionlist)
	DataXYSet=[]
	DataXSet=[]
	DataYSet=[]
	DataRSet=[]
	for l in range(Depth-1):
		DataXY = actionlist[:l+2]
		DataX = actionlist[:l+1]
		DataY = actionlist[l+1:l+2]
		DataR = effective_rank_F(actionlist = DataXY)
		DataXYSet.append(DataXY)
		DataXSet.append(DataX)
		DataYSet.append(DataY)
		DataRSet.append(DataR)
	return DataXSet, DataYSet, DataRSet,DataXYSet

def DataSet_app_fun(DataXSet,DataYSet, DataRSet,DataXYSet, Depth, actionlist_batch, effective_rank_F):
	Rew_batch = []
	for b in range(len(actionlist_batch)):
		actionlist = actionlist_batch[b]
		Depth  =  len(actionlist)
		Rew = effective_rank_F(actionlist = actionlist)
		Rew_batch.append(Rew)
		for l in range(Depth-1):
			DataXY = actionlist[:l+2]
			found = any( sublist  == DataXY for sublist in DataXYSet)
			# print("Found!" if found else "Not Found.")
			if not found:
				DataX = actionlist[:l+1]
				DataY = actionlist[l+1:l+2]
				DataR = effective_rank_F(actionlist = DataXY)
				DataXYSet.append(DataXY)
				DataXSet.append(DataX)
				DataYSet.append(DataY)
				DataRSet.append(DataR)
	return DataXSet, DataYSet, DataRSet,DataXYSet, Rew_batch



class DataSet_sampler(nn.Module):
	def __init__(self, agent, reward_F, Depth):
		super(DataSet_sampler, self).__init__()
		self.agent = agent
		self.reward_F = reward_F
		self.Depth = Depth
	
	def forward(self, DataXSet, DataYSet, DataRSet, DataXYSet, batch_size):
		actionlistAG_batch= actionlistAG_fun( Depth=self.Depth, batch_size=batch_size, agent=self.agent)
		DataXSet, DataYSet, DataRSet, DataXYSet, Rew_batch = DataSet_app_fun(DataXSet= DataXSet,DataYSet= DataYSet,DataRSet=DataRSet,DataXYSet= DataXYSet,  
														   		Depth= self.Depth, actionlist_batch=actionlistAG_batch, effective_rank_F=self.reward_F)
		return DataXSet, DataYSet, DataRSet, DataXYSet , Rew_batch



	

	
				

def fake_effective_rank_F(actionlist):
	rank = torch.tensor(np.random.randint(1, 20)).reshape(1)
	return rank.tolist()






if __name__ == '__main__':
	torch.set_default_dtype(torch.float64)
	
	Nq = 4
	Depth = 6	
	actionlistBW = actionlistBW_fun(  Nq=4, Depth=6)
	print("\nactionlistBW", actionlistBW)

	Nq = Nq
	Depth = Depth
	embed_dim = 10
	num_heads = 1
	TransformerClassifier = CustomTransformerClassifier(vocab_size = Nq*Nq+1, max_seq_len=Depth, embed_dim= embed_dim, num_heads=num_heads, ffn_hidden_dim=embed_dim*5, num_classes= Nq*Nq)
	actionlistAG_batch= actionlistAG_fun( Depth=Depth, batch_size=2, agent=TransformerClassifier)
	print("\nactionlistAG_batch", actionlistAG_batch)
	
	DataXSet, DataYSet, DataRSet,DataXYSet = DataSet_ini_fun(Depth= Depth, actionlist=actionlistBW, effective_rank_F=fake_effective_rank_F)
	print('\n',DataXYSet,'\n', DataRSet)

	DataXYSet=[]
	DataXSet=[]
	DataYSet=[]
	DataRSet=[]

	DataXSet, DataYSet, DataRSet,DataXYSet = DataSet_app_fun(DataXSet= DataXSet,DataYSet= DataYSet,DataRSet=DataRSet,DataXYSet= DataXYSet,  Depth= Depth,
														  		actionlist_batch=actionlistAG_batch, effective_rank_F=fake_effective_rank_F)
	print('\n',len(DataXYSet),'\n',DataXYSet,'\n', len(DataRSet),'\n', DataRSet)
	print('\n',len(DataXSet),'\n',DataXSet,'\n',len(DataYSet),'\n',DataYSet)
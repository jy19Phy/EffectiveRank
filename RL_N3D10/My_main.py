import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import math
import numpy as np
import random

from Mymonitor import time_string_fun

from MyFisher_statez0 import InputX_fun
from MyFisher_theta import *
from MyFisher_circuit import  wiresIDSet_fun, wiresIDSet0_ini_fun, measurement_circuit_fun, my_quantum_circuit_function
from MyFisher_Rank import effective_rank_class

from MyRL_actionList import actionlistBW_fun

from MyRL_attention import CustomTransformerClassifier
from MyRL_DataSet import DataSet_sampler,DataSet_ini_fun
from MyRL_train import train_agent_fun
from My_test import circuit_plot_fun


def Export(DataXYSet,DataRSet, round,Depth, Rew_batch):
	opt_index = torch.argmax( torch.tensor(DataRSet).reshape(-1) ).item()
	actionlist_opt = DataXYSet[opt_index]
	rank_opt  = DataRSet[opt_index]
	subDataXYRSet = [ DataXY+DataR for DataXY,DataR in zip(DataXYSet,DataRSet) if len(DataXY)==Depth]

	np.savetxt("./TrainAct/"+str(round)+"actr_"+str(len(DataRSet))+"_"+str(rank_opt)+".csv",np.array(subDataXYRSet),fmt='%d',delimiter=',')
	print('\nround',round,'\nactionlist_opt', actionlist_opt,'\nrand_opt',rank_opt )
	circuit_plot_fun(Nq,actionlist_opt)
	with open("./TrainRew/rew_batch"+str(len(Rew_batch))+".txt", "+a") as file:
			file.write(str(round))
			for i in range(len(Rew_batch)):
				file.write("\t"+str( Rew_batch[i][0]) )
			file.write("\n")
	return actionlist_opt, rank_opt


if __name__ == '__main__':
	torch.set_default_dtype(torch.float64)
	# system env
	statetime= time_string_fun()

	# circuit env
	Nq = 3
	Depth = 10
	batch_x = 20
	InputX = torch.tensor(np.load("./Data/rhoXNq"+str(Nq)+"batch"+str(batch_x)+".npy"))
	print(InputX.shape)
	measurement_scheme = measurement_circuit_fun
	effective_rank_F = effective_rank_class(Nq, InputX, measurement_circuit_fun)

	# Reinforcement learning evn
	Nq = Nq
	Depth = Depth
	embed_dim = 10
	num_heads = 1
	TransformerClassifier = CustomTransformerClassifier(vocab_size = Nq*Nq+1, max_seq_len=Depth, embed_dim= embed_dim, num_heads=num_heads, ffn_hidden_dim=embed_dim*5, num_classes= Nq*Nq)

	DataSetG= DataSet_sampler(agent=TransformerClassifier, reward_F=effective_rank_F, Depth=Depth )
	
	DataXYSet=[]
	DataXSet=[]
	DataYSet=[]
	DataRSet=[]
	# actionlistBW = actionlistBW_fun(  Nq=Nq, Depth=Depth)
	# DataXSet, DataYSet, DataRSet,DataXYSet = DataSet_ini_fun(Depth= Depth, actionlist=actionlistBW, effective_rank_F=effective_rank_F)

	batch_sampling = 10

	for round in range(5000):
		time_string_fun()
		DataXSet, DataYSet, DataRSet,DataXYSet, Rew_batch = DataSetG( DataXSet=DataXSet, DataYSet=DataYSet, DataRSet=DataRSet, DataXYSet = DataXYSet, batch_size= batch_sampling)
		time_string_fun()
		actionlist_opt , rank_opt  = Export(DataXYSet,DataRSet, round,Depth, Rew_batch)
		time_string_fun()
		loss = train_agent_fun(DataXSet=DataXSet, DataYSet=DataYSet, DataRSet=DataRSet, agent=TransformerClassifier)
		time_string_fun()
		with open("./TrainRew/loss.txt", "+a") as file:
			file.write(str(round)+"\t"+str(loss.detach().numpy())+"\n")
	print('\n',statetime)
	time_string_fun()


	
	


	
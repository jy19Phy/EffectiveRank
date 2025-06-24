from MyFisher_theta import paramsSet0_fun, paramsSet_ext_fun
import pennylane as qml
import matplotlib.pyplot as plt
import torch  
import numpy as np
from torch.nn  import functional as F 

from MyFisher_theta import uniform_rand_theta
from MyRL_ID import wiresID_from_actionID_fun

from MyFisher_circuit import  measurement_circuit_fun
from MyFisher_Rank import effective_rank_class

def my_action_gate_function(wiresID, params ):
	# print("wiresID", wiresID)
	# print("params", params)
	if wiresID[0]==wiresID[1]:
		qml.Rot(params[0],params[1],params[2],wires=wiresID[0])
	else:
		qml.CNOT(wires = wiresID)

def quantum_circuit_fun(Nq, wiresIDSet):
	Depth = 20
	paramsSet = uniform_rand_theta(Depth*3).reshape(Depth,3)
	paramsSet  = paramsSet.reshape(-1,3)
	for  l  in range(len(wiresIDSet)):
		wiresID = wiresIDSet[l]
		param = paramsSet[l]
		my_action_gate_function(wiresID, param)
	return qml.probs(wires=[i for i in range(Nq)])

def circuit_plot_fun(Nq, actionID):
	wiresID =  wiresID_from_actionID_fun(actionID ,Nq)
	dev = qml.device("default.qubit", wires=Nq)
	circuit_train = qml.QNode( quantum_circuit_fun, dev, interface="torch" )
	# print("wiresID=\t", wiresID)
	print('\n',qml.draw( circuit_train, decimals=None)(Nq, wiresID))
	return 0 

def opt_circuit_plot_fun(Nq,actionlist, effective_rank_F):
	rank = effective_rank_F(actionlist)
	print('\nactionlist',actionlist,'\nrank',rank)	
	circuit_plot_fun(Nq = Nq, actionID = actionlist)
	return 0 



if __name__ == '__main__':
	torch.set_default_dtype(torch.float64)

	# circuit env
	Nq = 3
	Depth = 10
	batch_x = 20
	InputX = torch.tensor(np.load("./Data/rhoXNq"+str(Nq)+"batch"+str(batch_x)+".npy"))
	print(InputX.shape)
	measurement_scheme = measurement_circuit_fun
	effective_rank_F = effective_rank_class(Nq, InputX, measurement_circuit_fun)

	# circuit plot
	Nq = Nq

	actionlist =[ 0,4	,8	,1,	5,	0,	4,	8,	1,	5]
	actionlist =[ 0,4	,8	,1,	5,	0,	4,	8,	]
	opt_circuit_plot_fun(Nq,actionlist, effective_rank_F)
	

	actionlist = [0	,4	,8	,1	,5	,0	,4,	7	,4	,8]
	opt_circuit_plot_fun(Nq,actionlist, effective_rank_F)
	
# import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
import math
from os import stat
import numpy as np
import torch 
import random
from torch import nn
from torch.nn  import functional as F 
from torch.autograd import Variable
import pennylane as qml

def arbitrary_density_matrix_circuit( Nq, rho_matrix):	
	wires = [i for i in range(Nq)]
	qml.QubitDensityMatrix(rho_matrix, wires)

def my_action_gate_function(wiresID, params ):
	# print("wiresID", wiresID)
	# print("params", params)
	if wiresID[0]==wiresID[1]:
		qml.Rot(params[0],params[1],params[2],wires=wiresID[0])
	else:
		qml.CNOT(wires = wiresID)

def my_quantum_circuit_function(Nq, rho_matrix, wiresIDSet, paramsSet ):
	# arbitrary_state_circuit(Nq, state_vector)
	arbitrary_density_matrix_circuit( Nq, rho_matrix)
	paramsSet  = paramsSet.reshape(-1,3)
	for  l  in range(len(wiresIDSet)):
		wiresID = wiresIDSet[l]
		param = paramsSet[l]
		my_action_gate_function(wiresID, param)
	return qml.probs(wires=[i for i in range(Nq)])

def measurement_circuit_fun(Nq, rho_matrix, wiresIDSet, paramsSet):
	# dev = qml.device("default.qubit", wires=Nq)
	dev = qml.device("default.mixed", wires= Nq)
	circuit_train = qml.QNode( my_quantum_circuit_function, dev, interface="torch" )
	P = circuit_train(Nq, rho_matrix, wiresIDSet, paramsSet)	
	# print(qml.draw(circuit_train)(Nq, rho_matrix, wiresIDSet, paramsSet))
	# print(qml.draw(circuit_train, decimals=None)(Nq, rho_matrix, wiresIDSet, paramsSet))
	return P	

def wiresID_ext_fun(wiresIDSet, Nq):
	wiresID = [random.randint(0, Nq-1), random.randint(0,Nq-1) ]
	# print(wiresID)
	wiresIDSet.append(wiresID)
	# print(wiresIDSet, len(wiresIDSet))
	return wiresIDSet

def wiresIDSet_fun(Depth, Nq):
	wiresIDSet =[ [0,0] ]
	for _ in range(Depth-1):
		wiresIDSet = wiresID_ext_fun(wiresIDSet, Nq)
	return wiresIDSet

def wiresIDSet0_ini_fun():
	wiresIDSet0 = [[0,0]]
	return wiresIDSet0


if __name__ == '__main__':

	num_qubit = 2
	Depth = 11
	
	batch_x = 20
	
	InputX = torch.tensor(np.load("./Data/rhoXNq"+str(num_qubit)+"batch"+str(batch_x)+".npy"))
	print(InputX.shape)



	Nq = num_qubit

	Depth = Nq*Nq
	wiresIDSet = wiresIDSet_fun(Depth, Nq)

	rho_matrix = InputX[0]


	paramsSet = torch.randn( (Depth,3), requires_grad=True )
	P = measurement_circuit_fun(Nq, rho_matrix, wiresIDSet, paramsSet)
	print(P)

	# gatetype = torch.tensor(wiresIDSet)[:,0]-torch.tensor(wiresIDSet)[:,1]
	# zero_positions = torch.where(gatetype == 0)[0]

	# # print(paramsSet.shape)
	# # print(gatetype)
	# print( len(zero_positions) )
	# # print(paramsSet[zero_positions])

	

	# dev = qml.device("default.qubit", wires=Nq)
	# circuit_train = qml.QNode( my_quantum_circuit_function, dev, interface="torch" )
	# P = circuit_train(Nq, state_vector, wiresIDSet, paramsSet)
	# # print(P)
	
	# print(qml.draw(circuit_train, decimals=None)(Nq, state_vector, wiresIDSet, paramsSet))
	# # paramsSet.grad = None  
	# # P[0].backward(retain_graph=True)
	# # gradRes= paramsSet.grad
	# # print(gradRes)
	# # print(gradRes[zero_positions])

	# # paramsSet.grad = None  
	# # P[1].backward(retain_graph=True)
	# # gradRes= paramsSet.grad
	# # print(gradRes[zero_positions])

	# LogP = torch.log(P)
	# print(LogP)


	


	
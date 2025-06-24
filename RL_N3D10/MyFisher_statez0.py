# import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
import pennylane as qml
import math
import numpy as np
import torch
import random

def random_sym_state_fun( Nq, Ndown,  Nd=2):
	WF = torch.rand(Nd**Nq)+torch.rand(Nd**Nq)*1.0j
	for i in range(Nd**Nq):
		if  bin(i).count('1') != Ndown:
			WF[i]= 0
	Nor= torch.real(torch.sum(torch.conj(WF)*WF))
	WFNor= WF/torch.sqrt(Nor)
	return WFNor


def InputX_fun(batch_x, Nq):
	if(Nq%2==1) :
		print("error with odd Nq")
	Ndown = Nq/2
	InputX = []
	for i in range( batch_x):
		state_vector = random_sym_state_fun(Nq = Nq, Ndown = Ndown)
		InputX.append(state_vector)
	return torch.cat(InputX).reshape(batch_x,-1)


def check_norm_fun( state_vector, Nq ):
	def arbitrary_state_circuit(state_vector,  Nq):
	# 使用 QubitStateVector 来嵌入包含相位信息的量子态
		wires = [i for i in range(Nq)]
		qml.QubitStateVector(state_vector, wires)
		return qml.state()
	dev = qml.device("default.qubit", wires= Nq, shots =None)
	circuit = qml.QNode( arbitrary_state_circuit, dev )
	result  = circuit(state_vector, Nq)
	Norm = torch.sum( torch.square( torch.abs( result)))
	print('Norm= ', Norm)
	return Norm

def check_Mz_fun(state_vector, Nq):
	def arbitrary_state_circuit(state_vector,  Nq):
	# 使用 QubitStateVector 来嵌入包含相位信息的量子态
		wires = [i for i in range(Nq)]
		qml.QubitStateVector(state_vector, wires)
		return [qml.expval(qml.PauliZ(i)) for i in range(Nq)]
	dev = qml.device("default.qubit", wires= Nq, shots =None)
	circuit = qml.QNode( arbitrary_state_circuit, dev )
	result  = circuit(state_vector, Nq)
	TotalMz = torch.sum(torch.tensor(result) )
	print("Final state:", state_vector)
	print("mz for each site:", torch.tensor(result) )
	print("Total mz:", TotalMz)
	return TotalMz

		




if __name__ == '__main__':
	torch.set_num_threads(1)

	Nq = 4 
	batch_x = 2
	InputX = InputX_fun(batch_x,Nq)
	print(InputX)

	for i in range(len(InputX)):
		check_norm_fun(InputX[i], Nq)
		check_Mz_fun(InputX[i], Nq)
	
	





		


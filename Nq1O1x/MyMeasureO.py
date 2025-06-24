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
from Mygate import *
from Myrho import *
from Mycircuit import rho_circuit_operation_fun

def measure_Ox_fun(Nq):
	U_x = 1./np.sqrt(2.)*torch.tensor([[1.0+0.0j, 1.0+0.0j],  [1.0+0.0j, -1+0.0j]]).reshape(2,2)
	measure_Ox = U_x 
	for q in range(1,Nq):
		measure_Ox = torch.kron( measure_Ox, U_x)
	return measure_Ox

def measure_Oy_fun(Nq):
	U_y = 1./np.sqrt(2.)*torch.tensor([[1.0+0.0j, 0.0+1.0j],  [1.0+0.0j, 0.0-1.0j]]).reshape(2,2)
	measure_O = U_y 
	for q in range(1,Nq):
		measure_O = torch.kron( measure_O, U_y)
	return measure_O


def measurementX_fun(rho, Nq, Nd=2):
	rho = rho.reshape(-1, Nd**Nq, Nd**Nq)
	measure_Ox = measure_Ox_fun(Nq).reshape(Nd**Nq, Nd**Nq)
	probabilities = torch.einsum('ik, bkj,ji-> bi',measure_Ox, rho, measure_Ox)
	# print("probabilities", probabilities)
	return probabilities

def measurementY_fun(rho, Nq, Nd=2):
	rho = rho.reshape(-1, Nd**Nq, Nd**Nq)
	measure_O = measure_Oy_fun(Nq).reshape(Nd**Nq, Nd**Nq)
	measure_Od = torch.conj( torch.transpose( measure_O, 0, 1 ))
	probabilities = torch.einsum('ik, bkj,ji-> bi',measure_O, rho, measure_Od)
	# print("probabilities", probabilities)
	return probabilities

def measurementZ_fun(rho, Nq, Nd=2):
	rho = rho.reshape(-1, Nd**Nq, Nd**Nq)
	probabilities = torch.einsum('bii-> bi',rho)
	# print("probabilities", probabilities)
	return probabilities


def measurementZ_circuit_fun( rho, param, Nq, Nd=2):
	rho = rho.reshape(-1, Nd**Nq, Nd**Nq)
	rho_new = rho_circuit_operation_fun( rho, param , Nq , Nd=2 )
	probabilities = measurementZ_fun(rho_new, Nq, Nd=2)
	return torch.real(probabilities)

def measurementX_circuit_fun( rho, param, Nq, Nd=2):
	rho = rho.reshape(-1, Nd**Nq, Nd**Nq)
	rho_new = rho_circuit_operation_fun( rho, param , Nq , Nd=2 )
	probabilities = measurementX_fun(rho_new, Nq, Nd=2)
	return torch.real(probabilities)

def measurementY_circuit_fun( rho, param, Nq, Nd=2):
	rho = rho.reshape(-1, Nd**Nq, Nd**Nq)
	rho_new = rho_circuit_operation_fun( rho, param , Nq , Nd=2 )
	probabilities = measurementY_fun(rho_new, Nq, Nd=2)
	return torch.real(probabilities)
	


if __name__ == '__main__':

	num_qubits = 1
	batch_rho = 1

	rho1 =random_densitymatrix_fun(batch=batch_rho,Nq=num_qubits)
	print(rho1)
	mz = measurementZ_fun(rho=rho1,Nq=num_qubits)
	mx = measurementX_fun(rho= rho1, Nq = num_qubits)
	my = measurementY_fun(rho= rho1, Nq = num_qubits)
	print('mz', mz)
	print('mx', mx)
	print('my', my)



	rho2 =random_dig_densitymatrix_fun(batch=batch_rho, Nq=num_qubits)
	print(rho2)
	mz = measurementZ_fun(rho=rho2,Nq=num_qubits)
	mx = measurementX_fun(rho= rho2, Nq = num_qubits)
	my = measurementY_fun(rho= rho2, Nq = num_qubits)
	print('mz', mz)
	print('mx', mx)
	print('my', my)
	rho3 =random_densitymatrix_fromDia_fun(batch=batch_rho, Nq= num_qubits)
	
	# num_U = 4
	# num_thetas = 3
	# param = torch.rand(num_U, num_thetas)


	# num_qubits = 2
	# batch_states = 2
	# state = random_stateSet_fun(batch= batch_states, Nq=num_qubits)
	# print(state.size())

	
	# state_circuit_operation_fun( state, param , num_qubits , Nd=2 )

	# P_state_new = measurement_circuit_fun(state=state, param=param, Nq=num_qubits) 
	# print(P_state_new)
	




	# state = random_stateSet_fun(batch=batch_states,  Nq=num_qubits )
	# state = one_state_fun(  Nq=num_qubits )
	# state = singlequbit_random_stateSet_fun(batch_states , Nq=1, Nd=2)
	











	



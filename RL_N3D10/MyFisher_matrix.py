# import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
import math
import numpy as np
from sympy import false
import torch 
import random
from torch import nn
from torch.nn  import functional as F 
from torch.autograd import Variable
from MyFisher_theta import *
from MyFisher_statez0 import InputX_fun
from MyFisher_circuit import  wiresIDSet_fun, measurement_circuit_fun


def grad_fun( LogP,  paramsSet, zero_positions):
		num_P = len(LogP)
		dP = []
		for i_p in range(num_P):
			paramsSet.grad = None  
			LogP[i_p].backward(retain_graph=True)
			gradRes= paramsSet.grad
			dPdtheta = gradRes.reshape(-1,3)[zero_positions]
			dP = dP + list(dPdtheta)
			del dPdtheta,gradRes
		# print("dP", dP) # num_P, num_thetas
		dP = torch.stack(dP).reshape(num_P, -1)
		# print("dP", dP) # num_P, num_thetas
		
		return dP

def dp_element_fun(InputX, measurement_circuit_fun, wiresIDSet, paramsSet,  Nq ):
	gatetype = torch.tensor(wiresIDSet)[:,0]-torch.tensor(wiresIDSet)[:,1]
	zero_positions = torch.where(gatetype == 0)[0]
	# print('zero_positions',zero_positions)
	num_P = 2**Nq 

	proPSet = []
	dPSet =[]
	for i in range(len(InputX)):
		state_vector = InputX[i]
		proP= measurement_circuit_fun(Nq, state_vector, wiresIDSet, paramsSet)
		LogP= torch.log(proP+1e-10)
		# print("P, LogP", proP, LogP)
		dP = grad_fun(  LogP = LogP, paramsSet=paramsSet,zero_positions = zero_positions)
		Value_dP = dP.detach().reshape(-1)
		Value_proP = proP.detach().reshape(-1)
		# print(dP.shape) # num_P, num_thetas
		del proP, LogP, dP
		proPSet = proPSet  + list(Value_proP)
		dPSet = dPSet +list( Value_dP)
	# print("proPSet", torch.tensor(proPSet) )
	proPSet = torch.tensor(proPSet).reshape(len(InputX), num_P)
	dPSet = torch.tensor(dPSet).reshape(len(InputX), num_P, -1 )
	return dPSet, proPSet


def fisher_matrix_fun(InputX, measurement_circuit_fun, wiresIDSet, paramsSet,  Nq ):
	dP, proP = dp_element_fun(InputX= InputX ,measurement_circuit_fun = measurement_circuit_fun,  wiresIDSet= wiresIDSet, paramsSet= paramsSet, Nq= Nq )
	# print("dP", dP.shape)
	batch_x, num_P, num_theta = dP.size()
	dP = dP.reshape(-1,num_theta)
	proP = proP.reshape(batch_x,num_P)
	fisher_matrix_xPtheta = torch.einsum( 'ik, il-> ikl', dP, torch.conj(dP) ).reshape(batch_x,num_P,-1)
	# print('fisher_maxtirx_xPtheta', fisher_matrix_xPtheta.dtype)
	# print('proP', proP.dtype)
	fisher_matrix_sumP = torch.einsum( 'ipj,ip-> ij', fisher_matrix_xPtheta, proP )
	del fisher_matrix_xPtheta
	fisher_matrix_element = fisher_matrix_sumP.reshape(batch_x,num_theta,num_theta)
	del fisher_matrix_sumP
	fisher_matrix_avgx = torch.mean(fisher_matrix_element, dim=0)
	fisher_matrix = fisher_matrix_avgx.reshape(num_theta, num_theta)
	del fisher_matrix_avgx
	# print("fisher_matrix")
	return fisher_matrix



	
if __name__ == '__main__':

	
	num_qubits = 2
	batch_x = 10

	InputX = InputX_fun(batch_x=batch_x, Nq= num_qubits)

	Depth = num_qubits*num_qubits
	wiresIDSet = wiresIDSet_fun(Depth = Depth, Nq= num_qubits)

	paramsSet = torch.randn( (Depth,3), requires_grad=True )


	fisher_matrix =fisher_matrix_fun(InputX = InputX, measurement_circuit_fun = measurement_circuit_fun, wiresIDSet= wiresIDSet, paramsSet= paramsSet,  Nq= num_qubits)



	
	














	



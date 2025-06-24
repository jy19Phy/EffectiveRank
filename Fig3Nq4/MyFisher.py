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
from Mytheta import *


def grad_fun( proP,  param):
	batch_x, num_P = proP.size()
	# print(batch_x, num_P)
	dP = []
	for i_x in range(batch_x):
		for i_p in range(num_P):
			proPtest = proP[i_x, i_p]
			# print("proPtest",  proPtest)
			# print("param", param)
			dPdtheta = torch.autograd.grad( outputs= (proPtest,) , inputs=(param, ), grad_outputs= torch.ones(proPtest.size()), 
									retain_graph= True, create_graph= True, only_inputs= True )
			dP = dP + list(dPdtheta)
			# print("dPdtheta", dPdtheta)
	dP = torch.cat(dP).reshape(batch_x, num_P, -1)
	# print(dP) # batch_x, num_P, num_param(num_thetas)
	return dP

def dp_element_fun(rho, param, Nq ,measurement_circuit_fun):
	proP= measurement_circuit_fun(rho, param, Nq)
	LogP= torch.log(proP)
	dP = grad_fun(  proP= LogP, param=param)
	Value_dP = dP.detach()
	Value_proP = proP.detach()
	# print(dP.shape) # batch_x, num_P, num_param(num_thetas)
	del proP, LogP, dP
	return Value_dP, Value_proP


def fisher_matrix_fun(rho, param, Nq ,measurement_circuit_fun):
	dP, proP = dp_element_fun(rho= rho , param = param, Nq = Nq , measurement_circuit_fun = measurement_circuit_fun)
	batch_x, num_P, num_theta = dP.size()
	dP = dP.reshape(-1,num_theta)
	proP = proP
	fisher_matrix_xPtheta = torch.einsum( 'ik, il-> ikl', dP, torch.conj(dP) ).reshape(batch_x,num_P,-1)
	# fisher_matrix_xPtheta = torch.einsum( 'ik, il-> ikl', dP, dP ).reshape(batch_x,num_P,-1)
	fisher_matrix_sumP = torch.einsum( 'ipj,ip-> ij', fisher_matrix_xPtheta, proP )
	fisher_matrix_element = fisher_matrix_sumP.reshape(batch_x,num_theta,num_theta)
	fisher_matrix_avgx = torch.mean(fisher_matrix_element, dim=0)
	fisher_matrix = fisher_matrix_avgx.reshape(num_theta, num_theta)
	return fisher_matrix	

def fisher_matrixSet_fun(num_qubits, batch_thetas, num_thetas, rho, measurement_scheme):
	fisher_matrixSet = []
	theta_Set =[]
	rank_FSet =[]
	for batch in range( batch_thetas):
		param = (normal_rand_theta(num_thetas)).requires_grad_(True)
		# print("theta param", param)
		# fisher_matrix=fisher_matrix_fun(state = state, param=param, Nq=num_qubits ,measurement_circuit_fun= measurement_circuit_fun)
		fisher_matrixOSet = []
		for measure_circuit_fun in measurement_scheme:
			fisher_matrixO=fisher_matrix_fun(rho = rho, param=param, Nq=num_qubits ,measurement_circuit_fun= measure_circuit_fun)
			fisher_matrixOSet.append( fisher_matrixO.unsqueeze(0))
		fisher_matrixOSet = torch.cat(fisher_matrixOSet)
		fisher_matrix = torch.mean(fisher_matrixOSet, dim=0)
		rank_F = torch.linalg.matrix_rank(fisher_matrix) 
		fisher_matrixSet.append(fisher_matrix.unsqueeze(0))
		theta_Set.append(param.unsqueeze(0))
		rank_FSet.append(rank_F.reshape(1,1))
		# print("theta:", param)
		# print("fisher:", fisher_matrix)
		# print("fisher rank:",torch.linalg.matrix_rank(fisher_matrix) )
	fisher_matrixSet = torch.cat( fisher_matrixSet)
	theta_Set = torch.cat(theta_Set)
	rank_FSet = torch.cat(rank_FSet )
	return fisher_matrixSet, rank_FSet

	
if __name__ == '__main__':

	
	num_qubits = 2
	num_thetas = 2
	batch_states = 100
	batch_thetas = 1000

	
	














	



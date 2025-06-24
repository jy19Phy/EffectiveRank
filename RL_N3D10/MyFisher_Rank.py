import torch 
import numpy as np
from MyFisher_matrix import fisher_matrix_fun
from MyFisher_theta import *
from MyFisher_statez0 import InputX_fun
from MyFisher_circuit import  wiresIDSet_fun, wiresID_ext_fun,  measurement_circuit_fun

from  MyRL_ID import wiresID_from_actionID_fun, actionIDSet_fun

def eig_fun(fisher):
	# print("eigE", torch.linalg.eigvals(fisher))
	if torch.isnan(fisher).any() or torch.isinf(fisher).any():
		print("输入矩阵包含非法值。")
		print(fisher)
	else:
		eigE = torch.real(torch.linalg.eigvals(fisher))
		eigE, _  = torch.sort( eigE, dim=-1)	
	return eigE

def effecive_rank_fun(InputX, measurement_circuit_fun, wiresIDSet, paramsSet,  Nq ):
	fisher_matrix =fisher_matrix_fun(InputX = InputX, measurement_circuit_fun = measurement_circuit_fun, wiresIDSet= wiresIDSet, paramsSet= paramsSet,  Nq= Nq)
	rank_F = torch.linalg.matrix_rank(fisher_matrix) 
	spectrum = eig_fun (fisher_matrix)
	num_theta = len(spectrum)
	# print("num_theta=" , num_theta)
	# print('eff_rank= ',rank_F)
	return rank_F, num_theta, spectrum

class effective_rank_class(nn.Module):
	def __init__(self,Nq, InputX, measurement_circuit_fun):
		super(effective_rank_class, self).__init__()
		self.Nq = Nq
		self.InputX= InputX
		self.measurement_circuit_fun = measurement_circuit_fun

	def forward(self, actionlist):
		wiresIDSet = wiresID_from_actionID_fun(actionlist, self.Nq )
		Depth = len(wiresIDSet)
		paramsSet = uniform_rand_theta(Depth*3).reshape(Depth,3)
		paramsSet.requires_grad_(True)
		fisher_matrix =fisher_matrix_fun(InputX = self.InputX, measurement_circuit_fun = self.measurement_circuit_fun, wiresIDSet= wiresIDSet, paramsSet= paramsSet,  Nq= self.Nq)
		rank_F = torch.linalg.matrix_rank(fisher_matrix) 
		spectrum = eig_fun (fisher_matrix)
		num_theta = len(spectrum)
		return rank_F.reshape(1).tolist()




if __name__ == '__main__':
	torch.set_default_dtype(torch.float64)

	num_qubit = 3
	Depth = 10
	batch_x = 20
	InputX = torch.tensor(np.load("./Data/rhoXNq"+str(num_qubit)+"batch"+str(20)+".npy"))
	print(InputX.shape)


	Depth = 1
	wiresIDSet = wiresIDSet_fun(Depth = Depth, Nq= num_qubit)
	paramsSet = uniform_rand_theta(len(wiresIDSet)*3).reshape(len(wiresIDSet),3)
	paramsSet.requires_grad_(True)
	# print("wiresIDSet=", wiresIDSet)
	# print("paramsSet=", paramsSet)

	for _ in range(30):	
		wiresIDSet = wiresID_ext_fun(wiresIDSet = wiresIDSet, Nq = num_qubit)
		paramsSet = paramsSet_ext_fun(paramsSet).detach()
		paramsSet.requires_grad_(True)

	print("wiresIDSet=", wiresIDSet)
	# print("paramsSet=", paramsSet)
		
	rank_F, num_theta, spectrum =effecive_rank_fun(InputX = InputX, measurement_circuit_fun = measurement_circuit_fun, wiresIDSet= wiresIDSet, paramsSet= paramsSet,  Nq= num_qubit)
	# print('num_theta =', num_theta)
	print('eff_rank= ',rank_F)
	# print('spectrum= ',spectrum)

	# wiresIDSet= [[0, 0], [3, 1], [2, 3], [3, 2]]
	# paramsSet= torch.tensor([[0.1383, 0.2392, 6.1401],
    #     [0.3039, 4.0405, 2.1098],
    #     [1.0624, 2.4434, 3.5286],
    #     [2.1745, 4.5414, 2.1447]], requires_grad=True)
	# rank_F, num_theta, spectrum =effecive_rank_fun(InputX = InputX, measurement_circuit_fun = measurement_circuit_fun, wiresIDSet= wiresIDSet, paramsSet= paramsSet,  Nq= num_qubits)
	# print('eff_rank= ',rank_F)
	
	Nq  = num_qubit
	InputX = InputX
	measurement_circuit_fun = measurement_circuit_fun
	effective_rank_F = effective_rank_class(Nq, InputX, measurement_circuit_fun)
	actionlist =  actionIDSet_fun(wiresIDSet, Nq)
	print('\nactionlist', actionlist)
	eff_r  = effective_rank_F(actionlist)
	print('\neff_r', eff_r)




	
	
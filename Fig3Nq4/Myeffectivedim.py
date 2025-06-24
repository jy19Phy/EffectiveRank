import torch 
import numpy as np


def normalized_fisher_matrix_fun( fisher_matrix_set):
	_ , num_thetas, _ = fisher_matrix_set.size()
	trF = torch.einsum('bii->b',fisher_matrix_set)
	mean_trF = torch.mean(trF)
	normalized_fisher_matrix = num_thetas/mean_trF*fisher_matrix_set
	return normalized_fisher_matrix  

def effectivedim_equation_fun(normalized_fisher_matrix, num_states):
	batch_thetas, num_thetas, _ = normalized_fisher_matrix.size()
	One = torch.eye(num_thetas).reshape(1,num_thetas,num_thetas).repeat(batch_thetas,1,1)
	nover2pi= num_states/np.pi/2.
	one_plus_F = One + nover2pi*normalized_fisher_matrix
	logdet = torch.linalg.slogdet(one_plus_F)[1]
	logx = logdet/2.
	Up = torch.logsumexp(logx, dim=0)-np.log(batch_thetas)
	effectivedim = 2.0* Up/ np.log(nover2pi)
	return effectivedim

def effctivedim_fun(fisher_matrixSet, num_states):
	normalized_fisher_matrix = normalized_fisher_matrix_fun(fisher_matrixSet)
	effectivedim= effectivedim_equation_fun(normalized_fisher_matrix= normalized_fisher_matrix, num_states= num_states)
	return effectivedim

if __name__ == '__main__':
	num_qubits = 1
	num_thetas = 3
	batch_states=5000

	state = torch.tensor(np.load("./Datastate/state1qubits5000batch.npy"))
	print(state.size())
	
	batch_thetas = 2
	# effectivedim = effctivedim_fun(num_qubits, num_thetas, batch_states, state, batch_thetas,  measurement_circuit_fun)
	
	# print(effectivedim)

	
	
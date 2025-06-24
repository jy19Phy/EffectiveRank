import torch 
from MyFisher import fisher_matrix_fun
from Mytheta import uniform_rand_theta

def eig_fun(fisher):
	# print("eigE", torch.linalg.eigvals(fisher))
	eigE = torch.real(torch.linalg.eigvals(fisher))
	eigE, _  = torch.sort( eigE, dim=-1)	
	return eigE

def spectrum_fun(state, param, Nq, measurement_circuit_fun):
	fisher_matrix = fisher_matrix_fun( state= state , param = param, Nq = Nq , measurement_circuit_fun = measurement_circuit_fun)
	spectrum = eig_fun (fisher_matrix)
	return spectrum


def mean_spectrum_fun(batch_thetas, num_thetas, num_qubits,  state, measurement_circuit_fun):
	spectrumSet = []
	for _ in range( batch_thetas):
		param = (uniform_rand_theta(num_thetas)).requires_grad_(True)
		spectrum=spectrum_fun(state=state, param=param, Nq=num_qubits, measurement_circuit_fun = measurement_circuit_fun)
		spectrumSet.append( torch.unsqueeze(spectrum, dim=0) )
	spectrumSet = torch.cat(spectrumSet, dim=0 )
	spectrum = torch.mean(spectrumSet.reshape(batch_thetas,num_thetas),dim = 0)
	return spectrum

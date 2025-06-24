import math
import stat
import numpy as np
import torch 
import random
from torch import nn
from torch.nn  import functional as F 
from torch.autograd import Variable
from Mygate import *
from Myrho import *
from Myeffectivedim import effctivedim_fun
from Mytheta import *
from MyFisher import *
from Myspectrum import eig_fun
from Mymonitor import get_max_memory_usage, time_string_fun

from MyMeasureO import  measurementX_circuit_fun, measurementY_circuit_fun, measurementZ_circuit_fun

def main_batchrho_fun(batch_rho, num_qubits):

	rho = random_densitymatrix_fromDia_fun( batch= batch_rho, Nq =num_qubits)
	np.save("./Data/rhoXNq"+str(num_qubits)+"batch"+str(batch_rho)+".npy", rho)
	np.savetxt("./Data/rhoXNq"+str(num_qubits)+"batch"+str(batch_rho)+".csv",rho)
	# batch_rho = 
	# rho = torch.tensor(np.load("./Data/xD1Nq"+str(num_qubits)+"batch"+str(batch_rho)+".npy"))
	print("state size = ", rho.size())
	
	batch_thetas = 100
	num_thetas = 4**num_qubits-1
	measurement_scheme = [measurementX_circuit_fun]
	fisher_matrixSet, rank_FSet = fisher_matrixSet_fun(num_qubits, batch_thetas, num_thetas,  rho, measurement_scheme)
	effdim = effctivedim_fun(fisher_matrixSet=fisher_matrixSet, num_states = batch_rho)

	np.save("ResX/Resv"+str(batch_rho)+"/matrixSet_Nq"+str(num_qubits)+".npy",fisher_matrixSet.detach().numpy())
	np.savetxt("ResX/Resv"+str(batch_rho)+"/matrixSet_Nq"+str(num_qubits)+".csv",fisher_matrixSet.reshape(batch_thetas, -1).detach().numpy(),delimiter=',')
	np.savetxt("ResX/Resv"+str(batch_rho)+"/effectivedim.txt", effdim.reshape(-1).numpy())
	np.savetxt("ResX/Resv"+str(batch_rho)+"/rank_FSet_Nq"+str(num_qubits)+".csv",rank_FSet.reshape(batch_thetas, -1).detach().numpy(),delimiter=',')

	spectrumSet = eig_fun(fisher_matrixSet)
	np.save("ResX/Resv"+str(batch_rho)+"/spectrumSet_Nq"+str(num_qubits)+".npy",spectrumSet.detach().numpy())
	np.savetxt("ResX/Resv"+str(batch_rho)+"/spectrumSet_Nq"+str(num_qubits)+".csv",spectrumSet.detach().numpy(), delimiter=',')
	mean_spectrum = torch.mean( spectrumSet, dim = 0 )
	np.savetxt("ResX/Resv"+str(batch_rho)+"/mean_spectrum.txt", mean_spectrum.reshape(-1).numpy())
	# print("mean_spectrum = ", mean_spectrum)

	print("effecitvedim = \t", effdim,"\t\t over paramter dimension = ", num_thetas)
	rank_F = torch.sum(rank_FSet)/batch_thetas
	print("rank = \t", torch.sum(rank_FSet)/batch_thetas)
	return num_thetas, effdim, rank_F
	



if __name__ == '__main__':
	# torch.set_default_dtype(torch.float64)
	time_string_fun()

	num_qubits = 1
	dSet = []
	for batch_rho in range(1,21,1):
		print('batch_rho =', batch_rho)
		num_thetas, deff, rank_F = main_batchrho_fun(batch_rho= batch_rho,  num_qubits= num_qubits)
		dSet.append(torch.tensor(batch_rho).reshape(-1))
		dSet.append(torch.tensor(num_thetas).reshape(-1))
		dSet.append(deff.reshape(-1))
		dSet.append(rank_F.reshape(-1) )
		get_max_memory_usage()
		dSetTemp = torch.cat(dSet).reshape(-1,4)
		np.savetxt("ResX/dSet.txt", dSetTemp.numpy())
	dSet = torch.cat(dSet).reshape(-1,4)
	np.savetxt("ResX/dSet.txt", dSet.numpy())

	time_string_fun()




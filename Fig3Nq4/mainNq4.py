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

def mainbolck_fun(num_block, num_qubits):
	num_Uhaar = num_qubits*num_block
	# batch_rho = 50
	# # rho1 =random_densitymatrix_fun(batch=batch_rho,Nq=num_qubits)
	# rho2 =random_dig_densitymatrix_fun(batch=batch_rho, Nq=num_qubits)
	# rho3 =random_densitymatrix_fromDia_fun(batch=batch_rho, Nq= num_qubits)
	# rho = torch.cat((rho2, rho3), dim=0) 
	# print("state size = ", rho.size())
	# batch_rho, _ = rho.size()
	# np.save("./Data/densitymatrix"+str(num_qubits)+"qubits"+str(batch_rho)+"batch.npy", rho)
	# np.savetxt("./Data/densitymatrix"+str(num_qubits)+"qubits"+str(batch_rho)+"batch.csv",rho)
	batch_rho = 100
	rho = torch.tensor(np.load("./Data/densitymatrix"+str(num_qubits)+"qubits"+str(batch_rho)+"batch.npy"))
	
	batch_thetas = 10
	num_thetas = num_Uhaar*3
	measurement_scheme = [measurementX_circuit_fun, measurementY_circuit_fun, measurementZ_circuit_fun]
	fisher_matrixSet, rank_FSet = fisher_matrixSet_fun(num_qubits, batch_thetas, num_thetas,  rho, measurement_scheme)
	effdim = effctivedim_fun(fisher_matrixSet=fisher_matrixSet, num_states = batch_rho)

	np.save("Resblock/Resv"+str(num_block)+"/matrixSet_Nq"+str(num_qubits)+".npy",fisher_matrixSet.detach().numpy())
	np.savetxt("Resblock/Resv"+str(num_block)+"/matrixSet_Nq"+str(num_qubits)+".csv",fisher_matrixSet.reshape(batch_thetas, -1).detach().numpy(),delimiter=',')
	np.savetxt("Resblock/Resv"+str(num_block)+"/effectivedim.txt", effdim.reshape(-1).numpy())
	np.savetxt("Resblock/Resv"+str(num_block)+"/rank_FSet_Nq"+str(num_qubits)+".csv",rank_FSet.reshape(batch_thetas, -1).detach().numpy(),delimiter=',')

	spectrumSet = eig_fun(fisher_matrixSet)
	np.save("Resblock/Resv"+str(num_block)+"/spectrumSet_Nq"+str(num_qubits)+".npy",spectrumSet.detach().numpy())
	np.savetxt("Resblock/Resv"+str(num_block)+"/spectrumSet_Nq"+str(num_qubits)+".csv",spectrumSet.detach().numpy(), delimiter=',')
	mean_spectrum = torch.mean( spectrumSet, dim = 0 )
	np.savetxt("Resblock/Resv"+str(num_block)+"/mean_spectrum.txt", mean_spectrum.reshape(-1).numpy())
	# print("mean_spectrum = ", mean_spectrum)

	rank_F = torch.sum(rank_FSet)/batch_thetas
	print("paramter dimension = ", num_thetas,"\t effdim = \t", effdim, "\t rank = \t",rank_F )
	return num_thetas, effdim, rank_F
	



if __name__ == '__main__':
	# torch.set_default_dtype(torch.float64)
	time_string_fun()

	num_qubits = 4
	dSet = []
	for block in range(15,100,500):
		print('block =', block)
		num_thetas, deff, rank_F = mainbolck_fun(num_block= block , num_qubits= num_qubits)
		dSet.append(torch.tensor(block).reshape(-1))
		dSet.append(torch.tensor(num_thetas).reshape(-1))
		dSet.append(deff.reshape(-1))
		dSet.append((deff/num_thetas).reshape(-1))
		dSet.append(rank_F.reshape(-1) )
		get_max_memory_usage()
		dSetTemp = torch.cat(dSet).reshape(-1,5)
		np.savetxt("Resblock/dSet.txt", dSetTemp.numpy())
	dSet = torch.cat(dSet).reshape(-1,5)
	np.savetxt("Resblock/dSet.txt", dSet.numpy())

	time_string_fun()




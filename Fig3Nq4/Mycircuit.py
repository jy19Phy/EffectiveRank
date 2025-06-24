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


def rho_circuit_operation_fun(  rho, param, Nq, Nd=2 ):
	num_haar, _ = param.reshape(-1, 3).size()
	num_block = int(num_haar/Nq)

	UhaarSet = haar_gate_fun(param ).reshape(-1,2,2)
	CNOT_brick_layer = CNOT_brick_layer_fun(Nq).reshape(Nd**Nq,Nd**Nq)
	rhoSet = rho.reshape(-1, Nd**Nq, Nd**Nq)

	for block in range(num_block):
		Uhaarlayer = UhaarSet[Nq*block]
		for q in range(1,Nq):
			Uhaarlayer = torch.kron(Uhaarlayer, UhaarSet[Nq*block+q])
		Uhaarlayer = torch.einsum('ij,jl-> il', CNOT_brick_layer, Uhaarlayer)
		Udagger = torch.conj( torch.transpose( Uhaarlayer, 0,1 ))
		rho_new = torch.einsum('ij,bjk,kl-> bil', Uhaarlayer, rhoSet, Udagger)
		rhoSet = rho_new
	return rho_new




if __name__ == '__main__':
	print('z')
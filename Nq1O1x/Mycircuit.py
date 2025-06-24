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
	rhoSet = rho.reshape(-1, Nd**Nq, Nd**Nq)
	Uhaar = haar_gate_fun(param).reshape(2,2)
	Udagger = torch.conj( torch.transpose( Uhaar, 0,1 ))
	rho_new = torch.einsum('ij,bjk,kl-> bil', Uhaar, rhoSet, Udagger)
	return rho_new

if __name__ == '__main__':
	Nq =2 
	param = torch.rand( 15)+0.j
	rho = generate_pure_density_matrix( batch= 1, Nq =Nq)
	rho_new = rho_circuit_operation_fun(rho, param, Nq)
	print(rho_new)
	print(torch.einsum( 'bii-> b',  rho_new.reshape(-1,4,4)))
	print(torch.einsum( 'bij, bji-> b',  rho_new.reshape(-1,4,4), rho_new.reshape(-1,4,4)) )
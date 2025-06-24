import math
import numpy as np
import torch 
import random
from torch import nn
from torch.nn  import functional as F 
from torch.autograd import Variable


def uniform_rand_theta(num_thetas):
	theta_min = 0
	theta_max = 2.*np.pi
	x = torch.rand(num_thetas)*(theta_max-theta_min)+theta_min
	return x

def normal_rand_theta(num_thetas):
	x = torch.randn(num_thetas)
	return x 

def paramsSet_ext_fun(paramsSet):
	l = len(paramsSet.reshape(-1,3))
	paramsSet = uniform_rand_theta((l+1)*3)
	# params = uniform_rand_theta(3).requires_grad_(True)
	# paramsSet = torch.cat( (paramsSet.reshape(-1), params), dim=0 )
	return paramsSet.requires_grad_(True)

def paramsSet0_fun( ):
	paramsSet = uniform_rand_theta(3).reshape(1,3)
	paramsSet.requires_grad_(True)
	return paramsSet
	


if __name__ == '__main__':

	Depth  = 3 
	paramsSet = uniform_rand_theta(Depth*3).reshape(Depth,3)
	paramsSet = paramsSet_ext_fun(paramsSet)
	print(paramsSet.shape)
	


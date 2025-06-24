import torch 
import numpy as np


def wiresIDSet_brickwall_fun(Nq):
	wiresIDSet =[]
	for q in range(Nq):
		wiresIDSet.append([q,q])
	for q in range(Nq-1):
		wiresIDSet.append([q,q+1])
	return wiresIDSet

def actionIDSet_fun(wiresIDSet, Nq):
	actionID = torch.tensor(wiresIDSet)[:,0]*Nq+torch.tensor(wiresIDSet)[:,1]
	return actionID.tolist()

def wiresID_from_actionID_fun(actionID, Nq):
	actionID = torch.tensor(actionID) 
	wiresIDSet0 = torch.div(actionID, Nq, rounding_mode='floor') 			# 取整除数
	wiresIDSet1=  torch.remainder(actionID, Nq)    		# 取余数
	wiresIDSet = torch.cat( (wiresIDSet0.reshape(-1,1), wiresIDSet1.reshape(-1,1)), dim= -1 )
	return wiresIDSet.tolist()



if __name__ == '__main__':

	num_qubit = 4

	batch_x = 2
	# InputX = InputX_fun(batch_x=batch_x, Nq= num_qubit)

	Depth = 5
	wiresIDSet =  torch.randint(0,num_qubit,(Depth,2)).tolist()
	print("wiresIDSet=", wiresIDSet)
	# paramsSet = uniform_rand_theta(len(wiresIDSet)*3).reshape(len(wiresIDSet),3)
	# paramsSet.requires_grad_(True)
	# print("wiresIDSet=", wiresIDSet)
	# print("paramsSet=", paramsSet)
	actionID = actionIDSet_fun(wiresIDSet, num_qubit)
	print("actionID", actionID)
	
	wiresIDSet = wiresID_from_actionID_fun(actionID, num_qubit)
	print("wiresIDSet=", wiresIDSet)
	

	
	



	
	
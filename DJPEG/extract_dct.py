import numpy as np
import math
import torch

#caclulate DCT basis
def cal(p,q):
	if p==0:
		ap = 1/(math.sqrt(8)) # NXN = 8X8
	else:
		ap = math.sqrt(0.25)
	if q==0:
		aq = 1/(math.sqrt(8))
	else:
		aq = math.sqrt(0.25)
	return ap,aq

def cal64(p,q):
	basis = np.zeros((8,8))
	ap,aq = cal(p,q)
	for m in range(0,8):
		for n in range(0,8):
			basis[m,n] = ap*aq*math.cos(math.pi*(2*m+1)*p/16)*math.cos(math.pi*(2*n+1)*q/16) # DCT formula
	return basis

def DCT_basis_64():
	basis_64 = np.zeros((8,8,64))
	idx = 0
	for i in range(8):
		for j in range(8):
			basis_64[:,:,idx] = cal64(i,j)
			idx = idx + 1
	return basis_64

def DCT_basis_torch():
    DCT_basis_64_ = DCT_basis_64()
    np_basis = np.zeros((64, 1, 8, 8)) #outchannel, inchannel, height, width
    for i in range(64):
        np_basis[i,0,:,:] = DCT_basis_64_[:,:,i]

    torch_basis = torch.from_numpy(np_basis).float()
    return torch_basis





def cal_barni(p,q):
	if p==0:
		ap = 1/(math.sqrt(16)) # NXN = 16X16
	else:
		ap = math.sqrt(0.125)
	if q==0:
		aq = 1/(math.sqrt(16))
	else:
		aq = math.sqrt(0.125)
	return ap,aq




def cal256_barni(p,q):
	basis_barni = np.zeros((16,16))
	ap,aq = cal_barni(p,q)
	for m in range(0,16):
		for n in range(0,16):
			basis_barni[m,n] = ap*aq*math.cos(math.pi*(2*m+1)*p/32)*math.cos(math.pi*(2*n+1)*q/32) # DCT formula
	return basis_barni

def DCT_basis_256_barni():
	basis_barni_256 = np.zeros((16,16,256))
	idx = 0
	for i in range(16):
		for j in range(16):
			basis_barni_256[:,:,idx] = cal256_barni(i,j)
			idx = idx + 1
	return basis_barni_256

def DCT_basis_torch_barni():
    DCT_basis_barni_256 = DCT_basis_256_barni()
    np_basis_barni = np.zeros((256, 1, 16, 16)) #outchannel, inchannel, height, width
    for i in range(256):
        np_basis_barni[i,0,:,:] = DCT_basis_barni_256[:,:,i]

    torch_basis_barni = torch.from_numpy(np_basis_barni).float()
    return torch_basis_barni


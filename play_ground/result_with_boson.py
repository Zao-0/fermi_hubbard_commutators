# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 16:58:45 2024

@author: God_Zao
"""

import numpy as np

fn = np.array([[0,0],[0,1]])
I2 = np.eye(2)
I4 = np.eye(4)
bn = np.diag([0,1,2,3])
tl = [1,np.sqrt(2),np.sqrt(3)]
b_up = np.diag(tl,k=-1)
b_dn = np.diag(tl,k=1)
ba = b_up+b_dn
bm = b_up-b_dn

def construct_op(op_list):
    result = np.eye(1)
    for c in op_list:
        if c== 'fn':
            result = np.kron(result,fn)
        elif c=='bn':
            result = np.kron(result,bn)
        elif c=='ba':
            result = np.kron(result,ba)
        elif c=='bm':
            result = np.kron(result,bm)
        elif c=='I2':
            result = np.kron(result,I2)
        elif c=='I4':
            result = np.kron(result,I4)
    return result

op1 = construct_op(['fn','I2','ba','I2','I2','I4'])
op2 = construct_op(['I2','fn','ba','I2','I2','I4'])
op3 = construct_op(['I2','I2','I4','fn','I2','ba'])
op4 = construct_op(['I2','I2','I4','I2','fn','ba'])
result1 = op1+op2+op3+op4
print(np.linalg.norm(result1,ord='fro'))


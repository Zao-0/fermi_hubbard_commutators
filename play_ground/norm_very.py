# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 14:30:54 2024

@author: God_Zao
"""

import numpy as np
from numpy.linalg import norm

I = np.eye(2)
M = np.random.rand(4, 4)

fro_norm_ref = norm(M,ord = 'fro')
fro_norm = norm(np.kron(M,I),ord = 'fro')

l2_norm_ref = norm(M,ord=2)
l2_norm = norm(np.kron(M,I),ord=2)
import numpy as np
from scipy import sparse
Id2 = sparse.csr_matrix(np.eye(2,dtype=np.float32))
Id4 = sparse.csr_matrix(np.eye(3,dtype=np.float32))

Z = sparse.csr_matrix(np.array([[1,0],[0,-1]],dtype = np.float32))

Fc = sparse.csr_matrix(np.array([[0,0],[1,0]],dtype= np.float32))
Fa = sparse.csr_matrix(np.array([[0,1],[0,0]],dtype= np.float32))

Ba = sparse.csr_matrix(np.diag([1.,np.sqrt(2)],k=1).astype(np.float32))
Bc = sparse.csr_matrix(np.diag([1.,np.sqrt(2)],k=-1).astype(np.float32))

n = sparse.csr_matrix(np.diag([0.,1.]).astype(np.float32))

operation_dict = {'Id2':Id2,
                  'Id4':Id4,
                  'Z':Z,
                  'Fc':Fc,
                  'Fa':Fa,
                  'Ba':Ba,
                  'Bc':Bc,
                  'ZFc':Z@Fc,
                  'ZFa':Z@Fa,
                  'n':n}


def get_op_from_list(opstr):
    r = np.eye(1)
    for op in opstr:
        r = sparse.kron(r,operation_dict[op])
    return r

def construct_op(op_list,N):
    result = sparse.csr_matrix(np.eye(12**N))
    vaccancy_site = ['Id2','Id2','Id4']
    innenu_site = ['Z','Id2','Id4']
    innend_site = ['Id2','Z','Id4']
    """
    u_site0 = [['ZFc','Z','Id4'],['Fa','Id2','Id4']]
    u_site1 = [['ZFa','Z','Id4'],['Fc','Id2','Id4']]
    d_site0 = [['Id2','ZFc','Id4'],['Z','Fa','Id4']]
    d_site1 = [['Id2','ZFa','Id4'],['Z','Fc','Id4']]
    """
    u_site0 = [['Fc','Z','Id4'],['ZFa','Id2','Id4']]
    u_site1 = [['Fa','Z','Id4'],['ZFc','Id2','Id4']]
    d_site0 = [['Id2','Fc','Id4'],['Z','ZFa','Id4']]
    d_site1 = [['Id2','Fa','Id4'],['Z','ZFc','Id4']]


    for op in op_list:
        temp = 0
        if op[0] == 'hu':
            l1 = vaccancy_site*op[1]+u_site0[0]+innenu_site*(op[2]-op[1]-1)+u_site0[1]+vaccancy_site*(N-1-op[2])
            temp1 = get_op_from_list(l1)
            l2 = vaccancy_site*op[1]+u_site1[0]+innenu_site*(op[2]-op[1]-1)+u_site1[1]+vaccancy_site*(N-1-op[2])
            temp2 = get_op_from_list(l2)
            temp = temp1+temp2
        elif op[0] == 'gu':
            l1 = vaccancy_site*op[1]+u_site0[0]+innenu_site*(op[2]-op[1]-1)+u_site0[1]+vaccancy_site*(N-1-op[2])
            temp1 = get_op_from_list(l1)
            l2 = vaccancy_site*op[1]+u_site1[0]+innenu_site*(op[2]-op[1]-1)+u_site1[1]+vaccancy_site*(N-1-op[2])
            temp2 = get_op_from_list(l2)
            temp = temp1-temp2
        elif op[0] == 'hd':
            l1 = vaccancy_site*op[1]+d_site0[0]+innend_site*(op[2]-op[1]-1)+d_site0[1]+vaccancy_site*(N-1-op[2])
            temp1 = get_op_from_list(l1)
            l2 = vaccancy_site*op[1]+d_site1[0]+innend_site*(op[2]-op[1]-1)+d_site1[1]+vaccancy_site*(N-1-op[2])
            temp2 = get_op_from_list(l2)
            temp = temp1-temp2
        elif op[0] == 'gd':
            l1 = vaccancy_site*op[1]+d_site0[0]+innend_site*(op[2]-op[1]-1)+d_site0[1]+vaccancy_site*(N-1-op[2])
            temp1 = get_op_from_list(l1)
            l2 = vaccancy_site*op[1]+d_site1[0]+innend_site*(op[2]-op[1]-1)+d_site1[1]+vaccancy_site*(N-1-op[2])
            temp2 = get_op_from_list(l2)
            temp = temp1-temp2
        elif op[0] == 'nu':
            l = vaccancy_site*op[1]+['n','Id2','Id4']+vaccancy_site*(N-op[1]-1)
            temp = get_op_from_list(l)
        elif op[0] == 'nd':
            l = vaccancy_site*op[1]+['Id2','n','Id4']+vaccancy_site*(N-op[1]-1)
            temp = get_op_from_list(l)
        elif op[0] == 'Ba':
            l1 = vaccancy_site*op[1]+['Id2','Id2','Bc']+vaccancy_site*(N-op[1]-1)
            temp1 = get_op_from_list(l1)
            l2 = vaccancy_site*op[1]+['Id2','Id2','Ba']+vaccancy_site*(N-op[1]-1)
            temp2 = get_op_from_list(l2)
            temp = temp1+temp2
        elif op[0] == 'Bm':
            l1 = vaccancy_site*op[1]+['Id2','Id2','Bc']+vaccancy_site*(N-op[1]-1)
            temp1 = get_op_from_list(l1)
            l2 = vaccancy_site*op[1]+['Id2','Id2','Ba']+vaccancy_site*(N-op[1]-1)
            temp2 = get_op_from_list(l2)
            temp = temp1-temp2
        result = result@temp
    return result
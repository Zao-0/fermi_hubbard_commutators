import math
import enum
from numbers import Rational
from collections.abc import Sequence
from functools import cache
import numpy as np
from scipy import sparse
from fh_comm.lattice import latt_coord_to_index, SubLattice


class FieldOpType(enum.Enum):
    """
    Fermionic field operator type.
    """
    FERMI_CREATE  = 0   # fermionic creation operator
    FERMI_ANNIHIL = 1   # fermionic annihilation operator
    FERMI_NUMBER  = 2   # fermionic/bosonic number operator
    FERMI_MODNUM = 3    # fermionic/bosonic number operator (modified) i.e. n_i-1/2

class FieldOpType_FB(enum.Enum):
    """
    Fermionic-Bosonic field operator type.
    """
    FERMI_CREATE  = 0   # fermionic creation operator
    FERMI_ANNIHIL = 1   # fermionic annihilation operator
    FERMI_NUMBER  = 2   # fermionic/bosonic number operator
    FERMI_MODNUM = 3    # fermionic/bosonic number operator (modified) i.e. n_i-1/2
    BOSON_CREATE = 4    # bosonic creation operator
    BOSON_ANNIHIL = 5   # bosonic annihilation operator
    BOSON_NUMBER = 6

class ElementaryFieldOp:
    """
    Elementary fermionic field operator.
    """
    def __init__(self, otype: FieldOpType, i: Sequence[int], s: int):
        self.otype = otype
        self.i = tuple(i)
        assert s in [0, 1]
        self.s = s

    def __str__(self) -> str:
        """
        Represent the operator as a string.
        """
        nt = 'up'
        if self.s==1:
            nt = 'dn'
        if self.otype == FieldOpType.FERMI_CREATE:
            return f"ad_{{{self.i}, {nt}}}"
        if self.otype == FieldOpType.FERMI_ANNIHIL:
            return f"a_{{{self.i}, {nt}}}"
        if self.otype == FieldOpType.FERMI_NUMBER:
            return f"n_{{{self.i}, {nt}}}"
        if self.otype == FieldOpType.FERMI_MODNUM:
            return f"mn_{{{self.i}, {nt}}}"
        assert False

class ElementaryFieldOp_FB(ElementaryFieldOp):
    def __init__(self, otype: FieldOpType_FB, i: Sequence[int], s: int):
        self.otype = otype
        self.i = tuple(i)
        assert s in [0, 1, 2]
        self.s = s
    
    def __str__(self) -> str:
        nt = 'up'
        if self.s==1:
            nt = 'dn'
        elif self.s==2:
            nt = 'bn'
        if self.otype == FieldOpType_FB.FERMI_CREATE:
            return f"ad_{{{self.i}, {nt}}}"
        if self.otype == FieldOpType_FB.FERMI_ANNIHIL:
            return f"a_{{{self.i}, {nt}}}"
        if self.otype == FieldOpType_FB.FERMI_NUMBER:
            return f"n_{{{self.i}, {nt}}}"
        if self.otype == FieldOpType_FB.FERMI_MODNUM:
            return f"mn_{{{self.i}, {nt}}}"
        if self.otype == FieldOpType_FB.BOSON_CREATE:
            return f"bc_{{{self.i}, {nt}}}"
        if self.otype == FieldOpType_FB.BOSON_ANNIHIL:
            return f"ba_{{{self.i}, {nt}}}"
        assert False

class ProductFieldOp:
    """
    Product of elementary fermionic field operators.
    """
    def __init__(self, ops: Sequence[ElementaryFieldOp], coeff: float):
        self.ops = list(ops)
        self.coeff = coeff

    def __rmul__(self, other):
        """
        Logical scalar product.
        """
        if not isinstance(other, (Rational, float)):
            raise ValueError("expecting a scalar argument")
        return ProductFieldOp(self.ops, other * self.coeff)

    def __matmul__(self, other):
        """
        Logical product.
        """
        return ProductFieldOp(self.ops + other.ops, self.coeff * other.coeff)

    def __neg__(self):
        """
        Logical negation.
        """
        return ProductFieldOp(self.ops, -self.coeff)

    def __str__(self) -> str:
        """
        Represent the operator as a string.
        """
        c = "" if self.coeff == 1 else f"({self.coeff}) "
        if not self.ops:
            # logical identity operator
            s = "id"
        else:
            s = ""
            for op in self.ops:
                s += ("" if s == "" else " ") + str(op)
        return c + s

class ProductFieldOp_FB(ProductFieldOp):
    """
    Product of elementary fermionic-bosonic field operators.
    """
    def __init__(self, ops: Sequence[ElementaryFieldOp_FB], coeff: float):
        self.ops = list(ops)
        self.coeff = coeff
    
    def __rmul__(self, other):
        """
        Logical scalar product.
        """
        if not isinstance(other, (Rational, float)):
            raise ValueError("expecting a scalar argument")
        return ProductFieldOp_FB(self.ops, other * self.coeff)
    
    def __matmul__(self, other):
        """
        Logical product.
        """
        return ProductFieldOp_FB(self.ops + other.ops, self.coeff * other.coeff)
    
    def __neg__(self):
        """
        Logical negation.
        """
        return ProductFieldOp_FB(self.ops, -self.coeff)
    
    def __str__(self) -> str:
        """
        Represent the operator as a string.
        """
        c = "" if self.coeff == 1 else f"({self.coeff}) "
        if not self.ops:
            # logical identity operator
            s = "id"
        else:
            s = ""
            for op in self.ops:
                s += ("" if s == "" else " ") + str(op)
        return c + s

class FieldOp:
    """
    Sum of products of fermionic field operators.
    """
    def __init__(self, terms: Sequence[ProductFieldOp]):
        self.terms = list(terms)

    def as_matrix(self, latt_shape: Sequence[int], translatt: SubLattice = None):
        """
        Generate the sparse matrix representation of the operator
        embedded in a square lattice with dimensions `latt_shape`
        and periodic boundary conditions.
        Optionally using shifted copies on sublattice `translatt`.
        """
        # number of lattice sites; factor 2 from spin
        L = 2 * math.prod(latt_shape)
        if not self.terms:
            # fast-return zero matrix if terms are empty
            return sparse.csr_matrix((2**L, 2**L))
        clist, alist, nlist, mlist = construct_fermionic_operators(L)
        mat = 0
        if translatt is None:
            tps = [len(latt_shape) * (0,)]
        else:
            tps = translatt.instantiate(len(latt_shape) * (0,), latt_shape)
        for tp in tps:
            for term in self.terms:
                fstring = sparse.identity(2**L)
                for op in term.ops:
                    # take spin into account for indexing
                    j = 2 * latt_coord_to_index(tuple(x + y for x, y in zip(op.i, tp)), latt_shape) + op.s
                    if op.otype == FieldOpType.FERMI_CREATE:
                        fstring = fstring @ clist[j]
                    elif op.otype == FieldOpType.FERMI_ANNIHIL:
                        fstring = fstring @ alist[j]
                    elif op.otype == FieldOpType.FERMI_NUMBER:
                        fstring = fstring @ nlist[j]
                    elif op.otype == FieldOpType.FERMI_MODNUM:
                        fstring = fstring @ mlist[j]
                    else:
                        raise RuntimeError(f"unexpected fermionic operator type {op.otype}")
                mat += float(term.coeff) * fstring
        return mat

    def support(self) -> list[tuple]:
        """
        Support of the operator: lattice sites which it acts on (including spin).
        """
        s = []
        for term in self.terms:
            for op in term.ops:
                s.append(op.i + (op.s,))
        return sorted(list(set(s)))

    def as_compact_matrix(self):
        """
        Generate the sparse matrix representation on a virtual lattice
        consisting of the sites acted on by the field operators.
        """
        if not self.terms:
            # fast-return zero matrix if terms are empty
            return sparse.csr_matrix((1, 1))
        # active sites (including spin)
        supp = self.support()
        L = len(supp)
        clist, alist, nlist, mlist = construct_fermionic_operators(L)
        # construct matrix representation
        mat = 0
        for term in self.terms:
            fstring = sparse.identity(2**L)
            for op in term.ops:
                # take spin into account for indexing
                j = supp.index(op.i + (op.s,))
                if op.otype == FieldOpType.FERMI_CREATE:
                    fstring = fstring @ clist[j]
                elif op.otype == FieldOpType.FERMI_ANNIHIL:
                    fstring = fstring @ alist[j]
                elif op.otype == FieldOpType.FERMI_NUMBER:
                    fstring = fstring @ nlist[j]
                elif op.otype == FieldOpType.FERMI_MODNUM:
                    fstring = fstring @ mlist[j]
                else:
                    raise RuntimeError(f"unexpected fermionic operator type {op.otype}")
            mat += float(term.coeff) * fstring
        return mat

    # TODO: th check whether this function need to be modified
    def quadratic_coefficients(self):
        r"""
        Find the coefficients in the representation
        :math:`\sum_{i,j} h_{ij} a^{\dagger}_i a_j`,
        assuming that the field operator actually has this form.
        """
        if not self.terms:
            # fast-return zero matrix if terms are empty
            return np.zeros((1, 1))
        # active sites (including spin)
        supp = self.support()
        L = len(supp)
        h = np.zeros((L, L))
        for term in self.terms:
            if len(term.ops) == 1:
                op = term.ops[0]
                if op.otype != FieldOpType.FERMI_NUMBER:
                    raise ValueError("expecting number operator")
                j = supp.index(op.i + (op.s,))
                h[j, j] += term.coeff
            elif len(term.ops) == 2:
                op_a = term.ops[0]
                op_b = term.ops[1]
                if op_a.otype != FieldOpType.FERMI_CREATE:
                    raise ValueError("expecting creation operator")
                if op_b.otype != FieldOpType.FERMI_ANNIHIL:
                    raise ValueError("expecting annihilation operator")
                i = supp.index(op_a.i + (op_a.s,))
                j = supp.index(op_b.i + (op_b.s,))
                h[i, j] += term.coeff
            else:
                raise ValueError("field operator not of expected form")
        return h

    def __add__(self, other):
        """
        Logical sum.
        """
        return FieldOp(self.terms + other.terms)

    def __sub__(self, other):
        """
        Logical difference.
        """
        return FieldOp(self.terms + [-term for term in other.terms])

    def __rmul__(self, other):
        """
        Logical scalar product.
        """
        if not isinstance(other, (Rational, float)):
            raise ValueError("expecting a scalar argument")
        return FieldOp([other * term for term in self.terms])

    def __matmul__(self, other):
        """
        Logical product.
        """
        # take all pairwise products
        return FieldOp([t1 @ t2 for t1 in self.terms for t2 in other.terms])

    def __str__(self) -> str:
        """
        Represent the operator as a string.
        """
        if not self.terms:
            # logical zero operator
            return "<empty FieldOp>"
        s = ""
        for term in self.terms:
            s += ("" if s == "" else " + ") + str(term)
        return s


class FieldOp_FB(FieldOp):
    """
    Sum of products of fermionic-bosonic field operators.
    """
    def __init__(self, terms: Sequence[ProductFieldOp_FB], N:int):
        self.terms = list(terms)
        assert N>1
        self.N = N
    
    def as_matrix(self, latt_shape: Sequence[int], translatt: SubLattice = None):
        # num of lattice sites
        # we consider the bosons as a 'Special' spin-status
        # Therefore, there factor 3 from spin (up, down, boson)
        L = 3* math.prod(latt_shape)
        if not self.terms:
            # fast-return zero matrix if terms are empty
            return sparse.csr_matrix((2**(2*latt_shape)*self.N**latt_shape, 2**(2*latt_shape)*self.N**latt_shape))
        clist, alist, nlist, mlist, bclist, balist = construct_Holstein_operators(L,N)
        mat = 0
        if translatt is None:
            tps = [len(latt_shape) * (0,)]
        else:
            tps = translatt.instantiate(len(latt_shape) * (0,), latt_shape)
        for tp in tps:
            for term in self.terms:
                fstring = sparse.identity(2**L)
                for op in term.ops:
                    # take spin into account for indexing
                    j = 3 * latt_coord_to_index(tuple(x + y for x, y in zip(op.i, tp)), latt_shape) + op.s
                    if op.otype == FieldOpType_FB.FERMI_CREATE:
                        fstring = fstring @ clist[j]
                    elif op.otype == FieldOpType_FB.FERMI_ANNIHIL:
                        fstring = fstring @ alist[j]
                    elif op.otype == FieldOpType_FB.FERMI_NUMBER:
                        fstring = fstring @ nlist[j]
                    elif op.otype == FieldOpType_FB.FERMI_MODNUM:
                        fstring = fstring @ mlist[j]
                    elif op.otype == FieldOpType_FB.BOSON_CREATE:
                        fstring = fstring @ bclist[j]
                    elif op.otype == FieldOpType_FB.BOSON_ANNIHIL:
                        fstring == fstring @ balist[j]
                    else:
                        raise RuntimeError(f"unexpected fermionic operator type {op.otype}")
                mat += float(term.coeff) * fstring
        return mat

    def support(self) -> list[tuple]:
        """
        Support of the operator: lattice sites which it acts on (including spin).
        """
        s = []
        for term in self.terms:
            for op in term.ops:
                s.append(op.i + (op.s,))
        return sorted(list(set(s)))
    
    def as_compact_matrix(self):
        """
        Generate the sparse matrix representation on a virtual lattice
        consisting of the sites acted on by the field operators.
        """
        if not self.terms:
            # fast-return zero matrix if terms are empty
            return sparse.csr_matrix((1, 1))
        # active sites (including spin)
        supp = self.support()
        modes_list = []
        for label in supp:
            if label[-1] == 2:
                modes_list.append('b') # b for Bosons
            else:
                modes_list.append('f') # f for Fermions
        L = len(supp)
        assert L == len(modes_list)
        modes_list = tuple(modes_list)
        clist, alist, nlist, mlist, bclist, balist = construct_Holstein_operators_alt(modes_list)
        # construct matrix representation
        mat = 0
        numf, numb = get_particles(modes_list)
        for term in self.terms:
            fstring = sparse.identity(2**numf*self.N**numb)
            for op in term.ops:
                # take spin into account for indexing
                j = supp.index(op.i + (op.s,))
                if op.otype == FieldOpType_FB.FERMI_CREATE:
                    fstring = fstring @ clist[j]
                elif op.otype == FieldOpType_FB.FERMI_ANNIHIL:
                    fstring = fstring @ alist[j]
                elif op.otype == FieldOpType_FB.FERMI_NUMBER:
                    fstring = fstring @ nlist[j]
                elif op.otype == FieldOpType_FB.FERMI_MODNUM:
                    fstring = fstring @ mlist[j]
                elif op.otype == FieldOpType_FB.BOSON_CREATE:
                    fstring = fstring @ bclist[j]
                elif op.otype == FieldOpType_FB.BOSON_ANNIHIL:
                    fstring == fstring @ balist[j]
                else:
                    raise RuntimeError(f"unexpected fermionic operator type {op.otype}")
            mat += float(term.coeff) * fstring
        return mat
    
    def quadratic_coefficients(self):
        r"""
        Find the coefficients in the representation
        :math:`\sum_{i,j} h_{ij} a^{\dagger}_i a_j`,
        assuming that the field operator actually has this form.
        """
        if not self.terms:
            # fast-return zero matrix if terms are empty
            return np.zeros((1, 1))
        # active sites (including spin)
        supp = self.support()
        L = len(supp)
        h = np.zeros((L, L))
        for term in self.terms:
            if len(term.ops) == 1:
                op = term.ops[0]
                if op.otype != FieldOpType_FB.FERMI_NUMBER:
                    raise ValueError("expecting number operator")
                j = supp.index(op.i + (op.s,))
                h[j, j] += term.coeff
            elif len(term.ops) == 2:
                op_a = term.ops[0]
                op_b = term.ops[1]
                if op_a.otype != FieldOpType_FB.FERMI_CREATE:
                    raise ValueError("expecting creation operator")
                if op_b.otype != FieldOpType_FB.FERMI_ANNIHIL:
                    raise ValueError("expecting annihilation operator")
                i = supp.index(op_a.i + (op_a.s,))
                j = supp.index(op_b.i + (op_b.s,))
                h[i, j] += term.coeff
            else:
                raise ValueError("field operator not of expected form")
        return h
    
    def __add__(self, other):
        """
        Logical sum.
        """
        return FieldOp_FB(self.terms + other.terms)
    
    def __sub__(self, other):
        """
        Logical difference.
        """
        return FieldOp_FB(self.terms + [-term for term in other.terms])
    
    def __rmul__(self, other):
        """
        Logical scalar product.
        """
        if not isinstance(other, (Rational, float)):
            raise ValueError("expecting a scalar argument")
        return FieldOp_FB([other * term for term in self.terms])
    
    def __matmul__(self, other):
        """
        Logical product.
        """
        # take all pairwise products
        return FieldOp_FB([t1 @ t2 for t1 in self.terms for t2 in other.terms])
    
    def __str__(self) -> str:
        """
        Represent the operator as a string.
        """
        if not self.terms:
            # logical zero operator
            return "<empty FieldOp_FB>"
        s = ""
        for term in self.terms:
            s += ("" if s == "" else " + ") + str(term)
        return s


@cache
def  construct_fermionic_operators(nmodes: int):
    """
    Generate sparse matrix representations of the fermionic creation and
    annihilation operators for `nmodes` modes (or sites),
    based on Jordan-Wigner transformation.
    """
    I = sparse.identity(2)
    Z = sparse.csr_matrix([[ 1.,  0.], [ 0., -1.]])
    U = sparse.csr_matrix([[ 0.,  0.], [ 1.,  0.]])
    clist = []
    mlist = [] # indeed 1/2 * -Z
    for i in range(nmodes):
        c = sparse.identity(1)
        m = sparse.identity(1)
        for j in range(nmodes):
            if j < i:
                c = sparse.kron(c, I)
                m = sparse.kron(m, I)
            elif j == i:
                c = sparse.kron(c, U)
                m = sparse.kron(m, Z)*(-.5)
            else:
                c = sparse.kron(c, Z)
                m = sparse.kron(m, I)
        c = sparse.csr_matrix(c)
        m = sparse.csr_matrix(m)
        c.eliminate_zeros()
        m.eliminate_zeros()
        clist.append(c)
        mlist.append(m)
    # corresponding annihilation operators
    alist = [sparse.csr_matrix(c.conj().T) for c in clist]
    # corresponding number operators
    nlist = []
    for i in range(nmodes):
        f = 1 << (nmodes - i - 1)
        data = [1. if (n & f == f) else 0. for n in range(2**nmodes)]
        nlist.append(sparse.dia_matrix((data, 0), 2*(2**nmodes,)))
    return clist, alist, nlist, mlist

@cache
def  construct_Holstein_operators(nmodes: int, boson_level:int):
    """
    Generate sparse matrix representations of the fermionic creation and
    annihilation operators for `nmodes` modes (or sites),
    based on Jordan-Wigner transformation.
    """
    assert nmodes%3 == 0, f'nmodes = {nmodes}'
    I = sparse.identity(2)
    Z = sparse.csr_matrix([[ 1.,  0.], [ 0., -1.]])
    U = sparse.csr_matrix([[ 0.,  0.], [ 1.,  0.]])
    BOSON_I = sparse.identity(boson_level)
    CREATOR = np.zeros((boson_level,boson_level))
    for i in range(1,boson_level):
        CREATOR[i][i-1] = np.sqrt(i)
    CREATOR = sparse.csr_matrix(CREATOR)
    clist = []
    mlist = [] # indeed 1/2 * -Z
    bclist = []
    for i in range(nmodes):
        c = sparse.identity(1)
        m = sparse.identity(1)
        bc = sparse.identity(1)
        for j in range(nmodes):
            if j < i:
                if j%3 ==2:
                    c = sparse.kron(c, BOSON_I)
                    m = sparse.kron(m, BOSON_I)
                    bc = sparse.kron(bc, BOSON_I)
                else:
                    c = sparse.kron(c, I)
                    m = sparse.kron(m, I)
                    bc = sparse.kron(bc, I)
            elif j == i:
                c = sparse.kron(c, U)
                m = sparse.kron(m, Z)*(-.5)
                bc = sparse.kron(bc, CREATOR)
            else:
                if j%3==2:
                    c = sparse.kron(c, I)
                    m = sparse.kron(m, BOSON_I)
                    bc = sparse.kron(bc, BOSON_I)
                else:
                    c = sparse.kron(c, Z)
                    m = sparse.kron(m, I)
                    bc = sparse.kron(bc, I)
        c = sparse.csr_matrix(c)
        m = sparse.csr_matrix(m)
        bc = sparse.csr_matrix(bc)
        c.eliminate_zeros()
        m.eliminate_zeros()
        bc.eliminate_zeros()
        clist.append(c)
        mlist.append(m)
        bclist.append(bc)
    # corresponding annihilation operators
    alist = [sparse.csr_matrix(c.conj().T) for c in clist]
    balist = [sparse.csr_matrix(bc.conj().T) for bc in bclist]
    # corresponding number operators
    nlist = []
    for i in range(nmodes):
        f = 1 << (nmodes - i - 1)
        data = [1. if (n & f == f) else 0. for n in range(2**nmodes)]
        nlist.append(sparse.dia_matrix((data, 0), 2*(2**nmodes,)))
    return clist, alist, nlist, mlist, bclist, balist

@cache
def construct_Holstein_operators_alt(nmodes_list, boson_level:int):
    nmodes_list = list(nmodes_list)
    I = sparse.identity(2)
    Z = sparse.csr_matrix([[ 1.,  0.], [ 0., -1.]])
    U = sparse.csr_matrix([[ 0.,  0.], [ 1.,  0.]])
    CREATOR = np.zeros((boson_level,boson_level))
    for i in range(1,boson_level):
        CREATOR[i][i-1] = np.sqrt(i)
    CREATOR = sparse.csr_matrix(CREATOR)
    BOSON_I = sparse.identity(boson_level)
    clist = []
    mlist = [] # indeed 1/2 * -Z
    bclist = []
    nmodes = len(nmodes_list)
    for i in range(nmodes):
        c = sparse.identity(1)
        m = sparse.identity(1)
        bc = sparse.identity(1)
        if nmodes_list[i] == 'b':
            for j in range(nmodes):
                if j==i:
                    bc = sparse.kron(bc, CREATOR)
                elif nmodes[j]=='b':
                    bc = sparse.kron(bc, BOSON_I)
                else:
                    bc = sparse.kron(bc,I)
        else:
            for j in range(nmodes):
                if nmodes[j]=='b':
                    c = sparse.kron(c,BOSON_I)
                    m = sparse.kron(m,BOSON_I)
                elif j<i:
                    c = sparse.kron(c,I)
                    m = sparse.kron(m,I)
                elif j>i:
                    c = sparse.kron(c,Z)
                    m = sparse.kron(m,I)
                else:
                    c = sparse.kron(c,U)
                    m = sparse.kron(m, Z)*(-.5)
        c = sparse.csr_matrix(c)
        m = sparse.csr_matrix(m)
        bc = sparse.csr_matrix(bc)
        c.eliminate_zeros()
        m.eliminate_zeros()
        bc.eliminate_zeros()
        clist.append(c)
        mlist.append(m)
        bclist.append(bc)
    alist = [sparse.csr_matrix(c.conj().T) for c in clist]
    balist = [sparse.csr_matrix(bc.conj().T) for bc in bclist]
    nlist = []
    for i in range(nmodes):
        f = 1 << (nmodes - i - 1)
        data = [1. if (n & f == f) else 0. for n in range(2**nmodes)]
        nlist.append(sparse.dia_matrix((data, 0), 2*(2**nmodes,)))
    return clist, alist, nlist, mlist, bclist, balist

def get_particles(modes_list):
    nf = 0
    nb = 0
    for c in modes_list:
        if c=='b':
            nb+=1
        else:
            nf+=1
    return nf, nb
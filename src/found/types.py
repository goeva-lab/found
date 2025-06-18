import numpy as np
import scipy.sparse as sp

BoolArr = np.ndarray[tuple[int], np.dtype[np.bool_]]
NumArr = np.ndarray[tuple[int], np.dtype[np.number]]

FloatMtx = np.ndarray[tuple[int, int], np.dtype[np.floating]]
IntMtx = np.ndarray[tuple[int, int], np.dtype[np.integer]]
SparseMtx = sp.csr_array | sp.csr_matrix | sp.csc_array | sp.csc_matrix

MatrixLike = FloatMtx | IntMtx | SparseMtx

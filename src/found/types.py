import numpy as np
import scipy.sparse as sp

BoolArr = np.ndarray[tuple[int], np.dtype[np.bool_]] | list[bool]
NumArr = np.ndarray[tuple[int], np.dtype[np.number]]

FloatMtx = np.ndarray[tuple[int, int], np.dtype[np.floating]]
IntMtx = np.ndarray[tuple[int, int], np.dtype[np.integer]]
SparseMtx = sp.csr_array | sp.csc_array

MatrixLike = FloatMtx | IntMtx | SparseMtx

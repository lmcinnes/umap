import numpy as np
import numba

@numba.njit()
def trustworthiness_vector_bulk(indices_source, indices_embedded, max_k):
    
    n_samples = indices_embedded.shape[0]
    trustworthiness = np.zeros(max_k + 1, dtype=np.float64)
    
    for i in range(n_samples):
        for j in range(max_k):
            
            rank = 0
            while indices_source[i, rank] != indices_embedded[i, j]:
                rank += 1
            
            for k in range(j + 1, max_k + 1):
                if rank > k:
                    trustworthiness[k] += rank - k
                    
    for k in range(1, max_k + 1):
        trustworthiness[k] = 1.0 - trustworthiness[k] * (2.0 / (n_samples * k *
                                (2.0 * n_samples - 3.0 * k -1.0)))
                                
    return trustworthiness

    

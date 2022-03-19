import math
import numpy as np

class Douglas:
    D = 3
    def sample(self, poly):
        mask = np.zeros((poly.shape[0],), dtype=int)
        mask[0] = 1
        endPoint = poly[0: 1, :] + poly[-1:, :]
        endPoint /= 2
        poly_append = np.concatenate([poly, endPoint], axis=0)
        self.compress(0, poly.shape[0], poly_append, mask)
        return mask

    def compress(self, idx1, idx2, poly, mask):
        p1 = poly[idx1, :]
        p2 = poly[idx2, :]
        A = (p1[1] - p2[1])
        B = (p2[0] - p1[0])
        C = (p1[0] * p2[1] - p2[0] * p1[1])

        m = idx1
        n = idx2
        if (n == m + 1):
            return
        d = abs(A * poly[m + 1: n, 0] + B * poly[m + 1: n, 1] + C) / math.sqrt(math.pow(A, 2) + math.pow(B, 2) + 1e-4)
        max_idx = np.argmax(d)
        dmax = d[max_idx]
        max_idx = max_idx + m + 1

        if dmax > self.D:
            mask[max_idx] = 1
            self.compress(idx1, max_idx, poly, mask)
            self.compress(max_idx, idx2, poly, mask)

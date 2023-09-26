import taichi as ti
from taichi.math import vec3, clamp
from utils import trilerp

@ti.data_oriented
class GridSDF:
    def __init__(self, lb, rt, res, sdf) -> None:
        self.lb = lb
        self.rt = rt
        self.res = res
        self.units = (rt - lb) / (res - 1)
        self.sdf = sdf

        self.density = ti.field(shape=(res, res, res), dtype=float)
        self.points = vec3.field(shape=(res, res, res))
        self.numeric_gradient = vec3.field(shape=(res, res, res))

    @ti.func
    def ijk2p(self, i, j, k):
        return self.lb + vec3(i,j,k) * self.units
    
    @ti.func
    def p2ijk(self, p):
        x = clamp(p, xmin=self.lb, xmax=self.rt-0.001)
        x -= self.lb
        x /= self.rt - self.lb
        x = x * (self.res - 1)
        x0 = ti.cast(x, int)
        return x0.x, x0.y, x0.z


    @ti.kernel
    def init_field(self):
        """
        Initialize a signed distance field.
        """
        for i, j, k in self.density:
            p = self.ijk2p(i, j, k)
            self.points[i,j,k] = p
            self.density[i,j,k] = self.sdf(p)

    @ti.kernel
    def calc_numeric_gradiant(self):
        """
        Calculate gradient using Sobel operator.

        Note:
        Seg fault is handled by taichi, outside being 0.
        """
        for i, j, k in self.numeric_gradient:
            gx = self.density[i+1, j-1, k-1] + 2*self.density[i+1, j, k-1] + self.density[i+1, j+1, k-1] \
                +2*self.density[i+1, j-1, k] + 4*self.density[i+1, j, k  ] + 2*self.density[i+1, j+1, k] \
                +self.density[i+1, j-1, k+1] + 2*self.density[i+1, j, k+1] + self.density[i+1, j+1, k+1] \
                -self.density[i-1, j-1, k-1] - 2*self.density[i-1, j, k-1] - self.density[i-1, j+1, k-1] \
                -2*self.density[i-1, j-1, k] - 4*self.density[i-1, j, k  ] - 2*self.density[i-1, j+1, k] \
                -self.density[i-1, j-1, k+1] - 2*self.density[i-1, j, k+1] - self.density[i-1, j+1, k+1]

            gy = self.density[i-1, j+1, k-1] + 2*self.density[i, j+1, k-1] + self.density[i+1, j+1, k-1] \
                +2*self.density[i-1, j+1, k] + 4*self.density[i, j+1, k  ] + 2*self.density[i+1, j+1, k] \
                +self.density[i-1, j+1, k+1] + 4*self.density[i, j+1, k+1] + self.density[i+1, j+1, k+1] \
                -self.density[i-1, j-1, k-1] - 2*self.density[i, j-1, k-1] - self.density[i+1, j-1, k-1] \
                -2*self.density[i-1, j-1, k] - 4*self.density[i, j-1, k  ] - 2*self.density[i+1, j-1, k] \
                -self.density[i-1, j-1, k+1] - 2*self.density[i, j-1, k+1] - self.density[i+1, j-1, k+1]

            gz = self.density[i-1, j-1, k+1] + 2*self.density[i-1, j, k+1] + self.density[i-1, j+1, k+1] \
                +2*self.density[i, j-1, k+1] + 4*self.density[i, j, k+1  ] + 2*self.density[i, j+1, k+1] \
                +self.density[i+1, j-1, k+1] + 2*self.density[i+1, j, k+1] + self.density[i+1, j+1, k+1] \
                -self.density[i-1, j-1, k-1] - 2*self.density[i-1, j, k-1] - self.density[i-1, j+1, k-1] \
                -2*self.density[i, j-1, k-1] - 4*self.density[i, j, k-1  ] - 2*self.density[i, j+1, k-1] \
                -self.density[i+1, j-1, k-1] - 2*self.density[i+1, j, k+1] - self.density[i+1, j+1, k+1]

            self.numeric_gradient[i,j,k] = vec3(gx, gy, gz).normalized()

    @ti.func
    def at(self, x):
        x = clamp(x, xmin=self.lb, xmax=self.rt-0.001)
        x -= self.lb
        x /= self.rt - self.lb
        x = x * (self.res - 1)
        x0 = ti.cast(x, int)
        x1 = x0 + 1
        density = trilerp(x, x0, x1,
                          self.density[x0[0], x0[1], x0[2]],
                          self.density[x1[0], x0[1], x0[2]],
                          self.density[x0[0], x1[1], x0[2]],
                          self.density[x1[0], x1[1], x0[2]],
                          self.density[x0[0], x0[1], x1[2]],
                          self.density[x1[0], x0[1], x1[2]],
                          self.density[x0[0], x1[1], x1[2]],
                          self.density[x1[0], x1[1], x1[2]])
        normal  = trilerp(x, x0, x1,
                          self.numeric_gradient[x0[0], x0[1], x0[2]],
                          self.numeric_gradient[x1[0], x0[1], x0[2]],
                          self.numeric_gradient[x0[0], x1[1], x0[2]],
                          self.numeric_gradient[x1[0], x1[1], x0[2]],
                          self.numeric_gradient[x0[0], x0[1], x1[2]],
                          self.numeric_gradient[x1[0], x0[1], x1[2]],
                          self.numeric_gradient[x0[0], x1[1], x1[2]],
                          self.numeric_gradient[x1[0], x1[1], x1[2]])

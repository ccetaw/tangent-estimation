import taichi as ti
from taichi.math import vec3, atan2, pi, sqrt, acos

# Define sdf of sphere
@ti.func
def sphere_sdf(o: vec3, r: float, p: vec3):
    """
    Compute sdf of a point for a sphere.

    args:
    - o: vec3. Origin of the sphere.
    - r: float. Radius of the sphere.
    - p: vec3. 3d point.

    returns:
    - d: float. Signed distane. Negative inside and positive outside.
    """
    return (p-o).norm() - r

@ti.func
def sphere_analytic_normal(o: vec3, p: vec3):
    """
    Analytical normal of a sphere.

    args:
    - o: vec3. Origin of the sphere.
    - p: vec3. 3d point.

    returns:
    - n: vec3. Normal.
    """
    return (p-o).normlaized()

@ti.func
def sphere_analytic_tangent_bitangent(o: vec3, p: vec3):
    """

    """
    n = sphere_analytic_normal(o, p)
    theta = atan2(n.y, n.x)
    if theta < 0:
        theta += 2*pi
    phi = acos(n.z)

    return vec3(ti.cos(theta)*ti.cos(phi), ti.cos(theta)*ti.sin(phi), -ti.sin(theta)), \
           vec3(-ti.sin(theta)*ti.sin(phi), ti.sin(theta)*ti.cos(phi), 0)

@ti.data_oriented
class GridSDF:

    def __init__(self, lb, rt, res) -> None:
        self.lb = lb
        self.rt = rt
        self.res = res
        self.units = (rt - lb) / (res - 1)

        self.density = ti.field(shape=(res, res, res), dtype=float)
        self.numeric_gradient = vec3.field(shape=(res, res, res), dtype=float)

    @ti.kernel
    def init_sphere_sdf(self, o: vec3, r: float):
        """
        Initialize a sphere sdf.
        """
        for i, j, k in self.density:
            p = self.lb + vec3(i,j,k) * self.units
            self.density[i,j,k] = sphere_sdf(o, r, p)

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
                +self.density[i+1, j-1, k+1] + 4*self.density[i+1, j, k+1] + self.density[i+1, j+1, k+1] \
                -self.density[i-1, j-1, k-1] + 2*self.density[i-1, j, k-1] + self.density[i-1, j+1, k-1] \
                -2*self.density[i-1, j-1, k] + 4*self.density[i-1, j, k  ] + 2*self.density[i-1, j+1, k] \
                -self.density[i-1, j-1, k+1] + 4*self.density[i-1, j, k+1] + self.density[i-1, j+1, k+1]

            gy = self.density[i-1, j+1, k-1] + 2*self.density[i, j+1, k-1] + self.density[i+1, j+1, k-1] \
                +2*self.density[i-1, j+1, k] + 4*self.density[i, j+1, k  ] + 2*self.density[i+1, j+1, k] \
                +self.density[i-1, j+1, k+1] + 4*self.density[i, j+1, k+1] + self.density[i+1, j+1, k+1] \
                -self.density[i-1, j-1, k-1] + 2*self.density[i, j-1, k-1] + self.density[i+1, j-1, k-1] \
                -2*self.density[i-1, j-1, k] + 4*self.density[i, j-1, k  ] + 2*self.density[i+1, j-1, k] \
                -self.density[i-1, j-1, k+1] + 4*self.density[i, j-1, k+1] + self.density[i+1, j-1, k+1]

            gz = self.density[i-1, j-1, k+1] + 2*self.density[i-1, j, k+1] + self.density[i-1, j+1, k+1] \
                +2*self.density[i, j-1, k+1] + 4*self.density[i, j, k+1  ] + 2*self.density[i, j+1, k+1] \
                +self.density[i+1, j-1, k+1] + 4*self.density[i+1, j, k+1] + self.density[i+1, j+1, k+1] \
                -self.density[i-1, j-1, k-1] + 2*self.density[i-1, j, k-1] + self.density[i-1, j+1, k-1] \
                -2*self.density[i, j-1, k-1] + 4*self.density[i, j, k-1  ] + 2*self.density[i, j+1, k-1] \
                -self.density[i+1, j-1, k-1] + 4*self.density[i+1, j, k+1] + self.density[i+1, j+1, k+1]

            self.numeric_gradient[i,j,k] = vec3(gx, gy, gz).normlaized()


o = vec3(0)
r = 1.0
res = 128
lb = vec3(-1.5, -1.5, -1.5)
rt = vec3(1.5, 1.5, 1.5)



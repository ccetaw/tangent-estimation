import taichi as ti
import numpy as np
from taichi.math import vec2, vec3, atan2, pi, acos, sin, cos, dot, sqrt, mat2, normalize, length
from grid_sdf import GridSDF
from utils import *

ti.init(arch=ti.gpu, kernel_profiler=True, device_memory_fraction=0.8)

# Grid SDF config
o = vec3(0.0)
r = 1.0
res = 256
lb = vec3(-1.5, -1.5, -1.5)
rt = vec3(1.5, 1.5, 1.5)
sdf = sd_sphere
grid_sdf = GridSDF(lb, rt, res, sdf)
grid_sdf.init_field()
grid_sdf.calc_numeric_gradiant()

# Cloud point shape representation
density = grid_sdf.density.to_numpy()
points = grid_sdf.points.to_numpy()
normals = grid_sdf.numeric_gradient.to_numpy()
surface_mask = np.abs(density) < 1e-3

surface_points = points[surface_mask]
surface_normals = normals[surface_mask]
n_points = surface_points.shape[0]

cloud = vec3.field(shape=n_points)
tmp_normals = vec3.field(shape=n_points)
cloud.from_numpy(surface_points)
tmp_normals.from_numpy(surface_normals)
cloud_normals = vec3.field(shape=n_points*2)

@ti.kernel
def init_cloud():
    for n in cloud:
        cloud_normals[2*n] = cloud[n]
        cloud_normals[2*n + 1] = cloud[n] + 0.05 * tmp_normals[n]
init_cloud()

# Sample points to show
n_samples = 20
points = vec3.field(shape=n_samples)
normals = vec3.field(shape=2 * n_samples)
tangents = vec3.field(shape=2 * n_samples)
bitangents = vec3.field(shape=2 * n_samples)
local_x = vec3.field(shape=2 * n_samples)
local_y = vec3.field(shape=2 * n_samples)

# Tangent estimation config
local_r = 0.1 * r
n_voxel = int(local_r / grid_sdf.units.min())
eps = 1e-2
neighbors = vec2.field(shape=(2*n_voxel+1, 2*n_voxel+1, 2*n_voxel+1), offset=(-n_voxel, -n_voxel, -n_voxel))


@ti.kernel
def init_render_sample():
    for n in range(n_samples):
        m = n * int(n_points/n_samples)
        points[n] = cloud[m]

        # taichi ggui convention. line start point
        normals[2*n] = cloud[m]
        tangents[2*n] = cloud[m]
        bitangents[2*n] = cloud[m]
        local_x[2*n] = cloud[m]
        local_y[2*n] = cloud[m]

        # taichi ggui convention. line end point
        normals[2*n+1] = normals[2*n] + 0.1 * r * tmp_normals[m]
        u, v = build_orthonomal_basis(tmp_normals[m])
        local_x[2*n+1] = local_x[2*n] + 0.1 * r * u
        local_y[2*n+1] = local_y[2*n] + 0.1 * r * v
        
        centroid = vec2(0)
        i0, j0, k0 = grid_sdf.p2ijk(points[n])
        for i in range(-n_voxel, n_voxel+1):
            for j in range(-n_voxel, n_voxel+1):
                for k in range(-n_voxel, n_voxel+1):
                    # select point having similar values
                    if grid_sdf.density[i0+i, j0+j, k0+k] - grid_sdf.density[i0, j0, k0] < eps:
                        p = grid_sdf.points[i0+i, j0+j, k0+k] - grid_sdf.points[i0, j0, k0]
                        # project to tangent plane
                        p = p - dot(p, tmp_normals[m]) * tmp_normals[m]
                        uv = vec2(dot(p, u), dot(p, v))
                        neighbors[i, j, k]= uv
                        centroid += uv
        centroid /= (2*n_voxel+1)**3
        C = mat2(0)
        for i in range(-n_voxel, n_voxel+1):
            for j in range(-n_voxel, n_voxel+1):
                for k in range(-n_voxel, n_voxel+1):
                    xi = neighbors[i, j, k] - centroid
                    weight = weight_func(length(xi)/local_r)
                    C[0, 0] = xi.x * xi.x 
                    C[0, 1] = xi.x * xi.y
                    C[1, 0] = xi.y * xi.x
                    C[1, 1] = xi.y * xi.y

        tr = C[0, 0] + C[1, 1]
        det = C[0, 0]*C[1, 1] - C[0, 1]*C[1, 0]
        lmd_1 = 0.5 * tr - sqrt(0.25 * tr * tr - det)
        lmd_2 = 0.5 * tr + sqrt(0.25 * tr * tr - det)
        LMD_1 = vec2(C[0, 1], lmd_1 - C[0, 0])
        LMD_2 = vec2(C[0, 1], lmd_2 - C[0, 0])

        tangents[2*n+1] = tangents[2*n] + 0.1 * r * normalize(LMD_1.x * u + LMD_1.y * v)
        bitangents[2*n+1] = bitangents[2*n] + 0.1 * r * normalize(LMD_2.x * u + LMD_2.y * v)



init_render_sample()

# @ti.kernel
# def init_render_sample():
#     for k in range(n_samples):
#         # fibonacci sphere
#         phi = pi * (3 - ti.sqrt(5.0))
#         z = 1.0 - (k / float(n_samples - 1)) * 2
#         radius = ti.sqrt(1 - z*z)
#         theta = phi * k
#         x = cos(theta) * radius
#         y = sin(theta) * radius
#         p = vec3(x, y, z)
#         points[k] = p
#         normals[2*k] = p
#         normals[2*k + 1] = p + 0.1 * r * sphere_analytic_normal(o, p)
#         tangents[2*k], bitangents[2*k] = p, p
#         tangent, bitangent = sphere_analytic_tangent_bitangent(o, p)
#         tangents[2*k+1], bitangents[2*k+1] = p + 0.1 * r * tangent, p + 0.1 * r * bitangent


# Render config
window = ti.ui.Window("Tangent and Bitangent Estimation", (1200, 1200), fps_limit=30)
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.Camera()
camera.position(3, 3, 3)
camera.up(0, 0, 1)
camera.lookat(o.x, o.y, o.z)
sphere = vec3.field(shape=1)
sphere[0] = o

x_axis = vec3.field(shape=2)
y_axis = vec3.field(shape=2)
z_axis = vec3.field(shape=2)

x_axis[0], y_axis[0], z_axis[0] = vec3(0), vec3(0), vec3(0)
x_axis[1], y_axis[1], z_axis[1] = vec3(3, 0, 0), vec3(0, 3, 0), vec3(0, 0 ,3)


t = 0
step = 0.001
while window.running:
    camera.position(3*sin(t), 3*cos(t), sin(t))
    scene.set_camera(camera)
    scene.ambient_light((0.8, 0.8, 0.8))
    scene.point_light(pos=camera.curr_position, color=(1, 1, 1))

    scene.particles(cloud, color = (0.5, 0.5, 0.5), radius = r/300)
    # scene.lines(cloud_normals, color = (0.9, 0.9, 0.1), width=3)
    scene.particles(points, color = (0.3, 0.6, 0.2), radius = r/100)
    
    scene.lines(x_axis, width=3, color=(0.1, 0.1, 0.9))
    scene.lines(y_axis, width=3, color=(0.1, 0.9, 0.1))
    scene.lines(z_axis, width=3, color=(0.9, 0.1, 0.1))
    scene.lines(normals, width=1, color=(0.8, 0.2, 0.4))
    scene.lines(local_x, width=1, color=(0.4, 0.8, 0.2))
    scene.lines(local_y, width=1, color=(0.2, 0.4, 0.8))
    scene.lines(tangents, width=1, color=(0.2, 0.8, 0.8))
    scene.lines(bitangents, width=1, color=(0.8, 0.2, 0.8))
    canvas.scene(scene)
    window.show()
    t += step

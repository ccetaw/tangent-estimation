import taichi as ti
from taichi.math import vec3, atan2, pi, acos, sin, cos
from grid_sdf import GridSDF
from utils import *

ti.init(arch=ti.gpu, kernel_profiler=True, device_memory_fraction=0.8)

# Grid SDF config
o = vec3(0.0)
r = 1.0
res = 128
lb = vec3(-1.5, -1.5, -1.5)
rt = vec3(1.5, 1.5, 1.5)
sdf = sd_bunny

grid_sdf = GridSDF(lb, rt, res, sdf)
grid_sdf.init_field()
grid_sdf.calc_numeric_gradiant()

# Cloud shape
cloud = vec3.field(shape=10 * res**2)
cloud_normal = vec3.field(shape=2*res**2)

@ti.kernel
def init_cloud():
    for i, j, k in grid_sdf.density:
        if ti.abs(grid_sdf.density[i, j, k]) < 1e-3:
            n = i * res**2 + j * res + k
            n = int(n/res)
            p = grid_sdf.ijk2p(i, j, k)
            cloud[n] = p
            cloud_normal[2*n] = p
            cloud_normal[2*n + 1] = p + 0.05 * grid_sdf.numeric_gradient[i, j, k]
init_cloud()

# Sample points to show
n_samples = 20
points = vec3.field(shape=n_samples)
normals = vec3.field(shape=2 * n_samples)
tangents = vec3.field(shape=2 * n_samples)
bitangents = vec3.field(shape=2 * n_samples)

@ti.kernel
def init_render_sample():
    for k in range(n_samples):
        # fibonacci sphere
        phi = pi * (3 - ti.sqrt(5.0))
        z = 1.0 - (k / float(n_samples - 1)) * 2
        radius = ti.sqrt(1 - z*z)
        theta = phi * k
        x = cos(theta) * radius
        y = sin(theta) * radius
        p = vec3(x, y, z)
        points[k] = p
        normals[2*k] = p
        normals[2*k + 1] = p + 0.1 * r * sphere_analytic_normal(o, p)
        tangents[2*k], bitangents[2*k] = p, p
        tangent, bitangent = sphere_analytic_tangent_bitangent(o, p)
        tangents[2*k+1], bitangents[2*k+1] = p + 0.1 * r * tangent, p + 0.1 * r * bitangent

init_render_sample()

# Render config
window = ti.ui.Window("Tangent and bitangent estimation", (1200, 1200), fps_limit=30)
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
step = 0.01
while window.running:
    camera.position(3*sin(t), 3*cos(t), 0)
    scene.set_camera(camera)
    scene.ambient_light((0.8, 0.8, 0.8))
    scene.point_light(pos=camera.curr_position, color=(1, 1, 1))

    scene.particles(cloud, color = (0.5, 0.5, 0.5), radius = r/300)
    scene.lines(cloud_normal, color = (0.9, 0.9, 0.1), width=3)
    # scene.particles(points, color = (0.3, 0.6, 0.2), radius = r/100)
    
    scene.lines(x_axis, width=3, color=(0.1, 0.1, 0.9))
    scene.lines(y_axis, width=3, color=(0.1, 0.9, 0.1))
    scene.lines(z_axis, width=3, color=(0.9, 0.1, 0.1))
    # scene.lines(normals, width=1, color=(0.9, 0.1, 0.1))
    # scene.lines(tangents, width=1, color=(0.1, 0.1, 0.9))
    # scene.lines(bitangents, width=1, color=(0.1, 0.9, 0.1))
    canvas.scene(scene)
    window.show()
    t += step

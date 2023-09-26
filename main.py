import taichi as ti
from taichi.math import vec3, atan2, pi, acos, sin, cos

@ti.func
def trilerp(x, x0, x1, q000, q100, q010, q110, q001, q101, q011, q111):
    """
    Trilinear interpolation.

    Input:
    - x: interpolate at x
    - x0, x1: 3D points in a right-handed system. In unit length, x0=(0,0,0), x1=(1,1,1)
    - q000, q010, ...: values at (0,0,0), (0,1,0), ...

    Output:
    - value at x
    """
    fx = (x1[0] - x[0]) / (x1[0] - x0[0])
    fy = (x1[1] - x[1]) / (x1[1] - x0[1])
    fz = (x1[2] - x[2]) / (x1[2] - x0[2])
    _fx = 1 - fx
    _fy = 1 - fy
    _fz = 1 - fz
    return ((q000 * fx + q100 * _fx) * fy + (q010 * fx + q110 * _fx) * _fy) * fz + ((q001 * fx + q101 * _fx) * fy + (q011 * fx + q111 * _fx) * _fy) * _fz

@ti.func
def sphere_analytic_normal(o, p):
    """
    Analytical normal of a sphere.

    args:
    - o: vec3. Origin of the sphere.
    - p: vec3. 3d point.

    returns:
    - n: vec3. Normal.
    """
    return (p-o).normalized()

@ti.func
def sphere_analytic_tangent_bitangent(o, p):
    """
    Analytical tangent and bitangent of a sphere.
    """
    n = sphere_analytic_normal(o, p)
    theta = atan2(n.y, n.x)
    if theta < 0:
        theta += 2*pi
    phi = acos(n.z)

    return vec3(cos(theta)*cos(phi), cos(theta)*sin(phi), -sin(theta)), vec3(-sin(phi), cos(phi), 0)
           

ti.init()

# Grid SDF config
o = vec3(0.0)
r = 1.0
res = 128
lb = vec3(-1.5, -1.5, -1.5)
rt = vec3(1.5, 1.5, 1.5)

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
window = ti.ui.Window("Tangent and bitangent estimation", (1200, 1200))
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


while window.running:
    camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.LMB)
    scene.set_camera(camera)
    scene.ambient_light((0.8, 0.8, 0.8))
    scene.point_light(pos=camera.curr_position, color=(1, 1, 1))

    scene.particles(sphere, color = (0.5, 0.5, 0.5), radius = r)
    scene.particles(points, color = (0.3, 0.6, 0.2), radius = r/100)
    scene.lines(x_axis, width=3, color=(0.1, 0.1, 0.9))
    scene.lines(y_axis, width=3, color=(0.1, 0.9, 0.1))
    scene.lines(z_axis, width=3, color=(0.9, 0.1, 0.1))
    scene.lines(normals, width=1, color=(0.9, 0.1, 0.1))
    scene.lines(tangents, width=1, color=(0.1, 0.1, 0.9))
    scene.lines(bitangents, width=1, color=(0.1, 0.9, 0.1))
    canvas.scene(scene)
    window.show()

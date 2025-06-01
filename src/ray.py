import taichi as ti

@ti.dataclass
class Ray:
    origin: ti.types.vector(3, ti.f32)
    direction: ti.types.vector(3, ti.f32)

@ti.func
def generate_camera_ray(camera, u: ti.f32, v: ti.f32) -> Ray:
    ray_origin = camera.get_pos()
    ray_dir = camera.get_ray_dir(u, v)
    return Ray(origin=ray_origin, direction=ray_dir)
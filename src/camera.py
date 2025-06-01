# camera.py
import taichi as ti
import numpy as np

@ti.data_oriented
class Camera:
    def __init__(self, res):
        self.fov = 60.0
        self.aspect_ratio = res[0] / res[1]

        self.radius = 600.0
        self.theta = 0.0
        self.phi = 1e-3

        self.pos = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.look_at = ti.Vector.field(3, dtype=ti.f32, shape=())

        self.set_position(np.array([0.0, 0.0, 0.0], dtype=np.float32))

    def set_position(self, center_point: np.ndarray):
        x = self.radius * ti.sin(self.phi) * ti.cos(self.theta)
        y = self.radius * ti.cos(self.phi)
        z = self.radius * ti.sin(self.phi) * ti.sin(self.theta)

        np_pos = np.array([x, y, z])
        np_pos = np_pos / np.linalg.norm(np_pos)
        np_pos *= self.radius
        np_pos += center_point

        self.pos.from_numpy(np_pos.astype(np.float32))
        self.look_at.from_numpy(center_point.astype(np.float32))

    def update_orbit(self, dtheta, dphi, dradius=0.0):
        self.theta += dtheta
        self.theta = self.theta % (2 * ti.math.pi)  # Wrap theta to [0, 2PI]
        self.phi = max(1e-3, min(self.phi + dphi, 3.13))  # Clamp phi to avoid poles
        self.radius = max(1.0, self.radius + dradius)     # Prevent flipping/negative


    @ti.func
    def get_ray_dir(self, u, v):
        theta = self.fov * ti.math.pi / 180.0
        half_height = ti.tan(theta / 2)
        half_width = self.aspect_ratio * half_height

        forward = (self.look_at[None] - self.pos[None]).normalized()
        up = ti.Vector([0.0, 1.0, 0.0])
        right = ti.Vector([1.0, 0.0, 0.0])
        if forward[1] != 1.0 and forward[1] != -1.0: # not parallel to up vector
            right = up.cross(forward).normalized()
        up = forward.cross(right).normalized()
        
        ray_dir = forward + (2.0 * u - 1.0) * half_width * right + (1.0 - 2.0 * v) * half_height * up
        
        return ray_dir.normalized()
    
    @ti.func
    def get_pos(self):
        return self.pos[None]
    
    def get_pos_py(self):
        return self.pos[None].to_numpy()
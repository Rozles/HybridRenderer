import taichi as ti
import numpy as np

@ti.dataclass
class PointLight:
    position: ti.types.vector(3, ti.f32)
    intensity: ti.types.vector(3, ti.f32)
    color: ti.types.vector(3, ti.f32)

@ti.data_oriented
class AreaLight:
    def __init__(self, center, radius, size):
        self.center = center
        self.radius = radius
        self.phi = 1e-3
        self.theta = 0.0
        self.position = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.normal = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.size = size
        self.intensity = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.color = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.position[None] = ti.Vector([0.0, 0.0, 0.0])
        self.normal[None] = ti.Vector([0.0, 1.0, 0.0])
        self.intensity[None] = ti.Vector([1.0, 1.0, 1.0])
        self.color[None] = ti.Vector([1.0, 1.0, 1.0])

    def set_light_properties(self, intensity, color):
        self.intensity.from_numpy(intensity.astype(np.float32))
        self.color.from_numpy(color.astype(np.float32))

    def update_orbit(self, dtheta, dphi):
        self.theta += dtheta
        self.theta = self.theta % (2 * ti.math.pi) 
        self.phi = max(1e-3, min(self.phi + dphi, 3.13))

        x = self.radius * ti.sin(self.phi) * ti.cos(self.theta)
        y = self.radius * ti.cos(self.phi)
        z = self.radius * ti.sin(self.phi) * ti.sin(self.theta)

        position = np.array([x, y, z], dtype=np.float32) + self.center
        normal = (self.center - position).astype(np.float32)
        normal /= np.linalg.norm(normal)

        self.position.from_numpy(position)
        self.normal.from_numpy(normal)

    @ti.func
    def sample_area_light(self, shading_point):
        u1 = ti.random()
        u2 = ti.random()
        r = self.size * ti.sqrt(u1)
        theta = 2.0 * ti.math.pi * u2
        x = r * ti.cos(theta)
        y = r * ti.sin(theta)

        n = self.normal[None]
        n = n.normalized()
        up = ti.Vector([0.0, 1.0, 0.0])
        if abs(n.dot(up)) > 0.999:
            up = ti.Vector([1.0, 0.0, 0.0])
        tangent = up.cross(n).normalized()
        bitangent = n.cross(tangent)

        light_sample_pos = self.position[None] + x * tangent + y * bitangent

        dir_to_light = light_sample_pos - shading_point
        dist = dir_to_light.norm()
        light_dir = dir_to_light.normalized()
        cos_theta = max(0.0, n.dot(-light_dir))

        area = ti.math.pi * self.size * self.size
        pdf = 1.0 / area

        return light_sample_pos, light_dir, dist, cos_theta, pdf
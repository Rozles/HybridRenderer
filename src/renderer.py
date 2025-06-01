import taichi as ti
from camera import Camera
from volume import Volume
from ray import generate_camera_ray, Ray
import numpy as np

from integrator import trace

@ti.data_oriented
class Renderer:
    def __init__(self, resolution, camera, volume, light):
        self.resolution = resolution
        self.camera = camera
        self.volume = volume
        self.light = light

        self.accum_image = ti.Vector.field(3, dtype=ti.f32, shape=resolution)
        self.display_image = ti.Vector.field(3, dtype=ti.f32, shape=resolution)

    @ti.kernel
    def render_one_sample(self, frame: int):
        for i, j in self.accum_image:
            u = (i + ti.random()) / self.resolution[0]
            v = (j + ti.random()) / self.resolution[1]

            ray = generate_camera_ray(self.camera, u, v)
            color = trace(ray, self.volume, self.light)

            old = self.accum_image[i, j]
            self.accum_image[i, j] = (old * frame + color) / (frame + 1)
            self.display_image[i, j] = self.accum_image[i, j]

    @ti.kernel
    def clear_image(self):
        for i, j in self.accum_image:
            self.accum_image[i, j] = ti.Vector([0.0, 0.0, 0.0])
            self.display_image[i, j] = ti.Vector([0.0, 0.0, 0.0])

    def save_image(self, filename):
        img_np = self.display_image.to_numpy()
        # Clamp and convert to uint8
        img_clamped = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
        from PIL import Image
        Image.fromarray(img_clamped).save(filename)
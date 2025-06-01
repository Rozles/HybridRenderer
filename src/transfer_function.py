import numpy as np
import taichi as ti

N_TF = 256

tf_color = ti.Vector.field(3, dtype=ti.f32, shape=N_TF)
tf_opacity = ti.field(dtype=ti.f32, shape=N_TF)
#tf_absorption = ti.field(dtype=ti.f32, shape=N_TF)
#tf_scattering = ti.field(dtype=ti.f32, shape=N_TF)
tf_emission = ti.Vector.field(3, ti.f32, shape=N_TF)
tf_roughness = ti.field(dtype=ti.f32, shape=N_TF)
tf_specular = ti.Vector.field(3, ti.f32, shape=N_TF)
tf_phase_g = ti.field(dtype=ti.f32, shape=N_TF)

def load_simple_transfer_function():
    color_transfer_function()

    for i in range(N_TF):
        t = i / (N_TF - 1)
        #color = np.array([t, t * 0.8, t * 0.6], dtype=np.float32)
        opacity = t
        #absorption = (1.0 - t) * 0.25
        #scattering = t
        emission = np.array([0.0, 0.0, t**4], dtype=np.float32)
        roughness = 1.0 - t 
        g = 1.0 - t * 1.5

        #tf_color[i] = color
        tf_opacity[i] = opacity
        #tf_absorption[i] = absorption
        #tf_scattering[i] = scattering
        tf_emission[i] = emission
        tf_roughness[i] = roughness
        tf_specular[i] = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        tf_phase_g[i] = g

def color_transfer_function():
    class Triangle:
        def __init__(self, start, peak, end, max_value, channel):
            self.start = start
            self.peak = peak
            self.end = end
            self.max_value = max_value
            self.channel = channel

    triangles = []
    triangles.append(Triangle(0.0, 0.05, 1.0, 0.25, 0))  # Red
    triangles.append(Triangle(0.0, 0.225, 1.0, 0.7, 1))  # Green
    triangles.append(Triangle(0.2, 0.26, 0.4, 1.0, 2))  # Blue

    for i in range(N_TF):
        t = i / (N_TF - 1)

        color = np.zeros(3, dtype=np.float32)
        for triangle in triangles:
            if t < triangle.start or t > triangle.end:
                value = 0.0
            elif t < triangle.peak:
                value = np.interp(t, [triangle.start, triangle.peak], [0.0, triangle.max_value])
            else:
                value = np.interp(t, [triangle.peak, triangle.end], [triangle.max_value, 0.0])
        
            color[triangle.channel] += value

        color = np.clip(color, 0.0, 1.0)
        tf_color[i] = color






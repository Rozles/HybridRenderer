# volume.py
import taichi as ti
import numpy as np
import SimpleITK as sitk

class Volume:
    def __init__(self, filename):
        self.data, self.origin, self.spacing = self.load_volume(filename)
        self.shape = self.data.shape

        self.center = (self.origin + np.array(self.shape) * self.spacing) / 2.0

        self.data_min = np.min(self.data)
        self.data_max = np.max(self.data)
        if self.data_max != self.data_min:
            self.data = (self.data - self.data_min).astype(np.float32) / (self.data_max - self.data_min)
        else:
            self.data = np.zeros_like(self.data)

        self.volume = None
        self.volume_origin = None
        self.volume_spacing = None

    def load_volume(self, filename):
        itkimage = sitk.ReadImage(filename)
        np_array = sitk.GetArrayFromImage(itkimage)
        origin = np.array(list(reversed(itkimage.GetOrigin())))
        spacing = np.array(list(reversed(itkimage.GetSpacing())))
        return np_array, origin, spacing

    def upload_to_taichi(self):
        self.volume = ti.field(dtype=ti.f32, shape=self.shape)
        self.volume.from_numpy(self.data.astype(np.float32))

        self.volume_origin = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.volume_spacing = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.volume_origin[None] = self.origin.astype(np.float32)
        self.volume_spacing[None] = self.spacing.astype(np.float32)


    @ti.func
    def inside_volume(self, v):
        inside = True
        for i in ti.static(range(3)):
            inside = inside and (0 <= v[i] < self.volume.shape[i])
        return inside
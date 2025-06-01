import taichi as ti
ti.init(arch=ti.gpu)

import time
from volume import Volume
from renderer import Renderer
from camera import Camera
from transfer_function import load_simple_transfer_function
from light import AreaLight
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from PIL import Image

def main():
    v = Volume("data/backpack_small.mhd")
    v.upload_to_taichi()

    resolution = (1024, 1024)

    camera = Camera(resolution)
    camera.set_position(v.center)

    light_pos = ti.Vector([600.0, 600.0, 600.0])
    light = AreaLight(v.center, radius=1000.0, size=200.0)

    light.set_light_properties(intensity=np.array([100.0, 100.0, 100.0], dtype=np.float32),
                            color=np.array([1.0, 1.0, 1.0], dtype=np.float32))

    light.update_orbit(ti.math.pi / 4, ti.math.pi / 4) 

    load_simple_transfer_function()

    gui = ti.GUI("Hybrid Renderer", resolution)
    frame = 0
    last_mouse_pos = None

    renderer = Renderer(resolution, camera, v, light)

    # PERFORMACE MEASUREMENT
    start_time = time.time()
    frame_times = []
    # similarities = []
    # reference_image = None
    # try:
    #     reference_image = np.array(Image.open("convergence_snapshot.png")) / 255.0
    #     reference_image = np.clip(reference_image, 0, 1)
    # except FileNotFoundError:
    #     print("Reference image not found, convergence plot will not be generated.")
    # ======================

    while gui.running:
        # PERFORMACE MEASUREMENT
        now = time.time()
        elapsed = now - start_time

        if frame == 3600:
            renderer.save_image("primer_backpack_1024.png")
            avg_frame_time = sum(frame_times) / len(frame_times)
            avg_fps = len(frame_times) / elapsed
            print(f"Average FPS over 60s: {avg_fps:.2f}")
            print(f"Average frame time: {avg_frame_time * 1000:.2f} ms")
            print(f"Total frames rendered: {len(frame_times)}")
            break

        # if frame == 3600:
        #     break

        frame_start = time.time()
        # ======================


        # camera orbit
        if gui.is_pressed(ti.GUI.LMB):
            if last_mouse_pos is not None:
                senzitivity = 5.0 
                dx = gui.get_cursor_pos()[0] - last_mouse_pos[0]
                dy = gui.get_cursor_pos()[1] - last_mouse_pos[1]
                dx *= senzitivity
                dy *= senzitivity
                camera.update_orbit(dtheta=dx, dphi=dy)  # Adjust sensitivity
                camera.set_position(v.center)
                light.update_orbit(dtheta=dx, dphi=dy)
                renderer.clear_image()
                frame = 0  # reset accumulation
            last_mouse_pos = gui.get_cursor_pos()
        else:
            last_mouse_pos = None

        # camera zoom
        if gui.get_event(ti.GUI.WHEEL):
            delta = gui.event.delta[1]
            camera.update_orbit(dtheta=0, dphi=0, dradius=-delta * 0.1)
            camera.set_position(v.center)
            renderer.clear_image()  # reset accumulation on zoom
            frame = 0


        renderer.render_one_sample(frame)
        gui.set_image(renderer.display_image)
        gui.show()

        frame += 1
        # PERFORMACE MEASUREMENT
        # if reference_image is not None:
        #     current_image = renderer.display_image.to_numpy()
        #     current_image = np.clip(current_image, 0, 1)
        #     sim = ssim(reference_image, current_image, channel_axis=-1, data_range=1.0)
        #     similarities.append(sim)

        frame_times.append(time.time() - frame_start)
        # ======================

    # PERFORMACE MEASUREMENT
    # if similarities:
    #     plt.figure()
    #     plt.plot(similarities)
    #     plt.xlabel("Frame")
    #     plt.ylabel("SSIM to reference")
    #     plt.title("Convergence Rate")
    #     plt.grid()
    #     plt.savefig("results/convergence_plot.png")
    #     plt.show()
    # ======================

def plot_volume_density():
    volume = Volume("data/artifix_small.mhd")
    density = volume.data.flatten()
    plt.figure(figsize=(10, 5))
    plt.hist(density, bins=256, range=(0, 1), density=True, alpha=0.7, color='blue')
    plt.title('Volume Density Distribution')
    plt.xlabel('Density Value')
    plt.ylabel('Frequency')
    plt.xlim(0, 1)
    plt.grid()
    plt.show()

if __name__ == "__main__":
    #plot_volume_density()
    main()
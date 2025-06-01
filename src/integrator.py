import taichi as ti
from ray import Ray
from transfer_function import tf_color, tf_opacity, tf_emission, tf_phase_g, N_TF
from light import AreaLight
from volume import Volume

@ti.func
def trace(ray: ti.template(), volume: ti.template(), light: ti.template()):
    throughput = ti.Vector([1.0, 1.0, 1.0])
    radiance = ti.Vector([0.0, 0.0, 0.0])
    ambient_light = ti.Vector([0.2, 0.2, 0.2]) 

    ray_origin = ray.origin
    ray_dir = ray.direction

    max_t = 1000.0

    max_bounces = 8
    for bounce in range(max_bounces):
        t = 0.0
        sigma_max = 1.0
        hit = False
        inside_volume = False

        x = ti.Vector([0.0, 0.0, 0.0])
        opacity = 0.0
        color = ti.Vector([0.0, 0.0, 0.0])
        emission = ti.Vector([0.0, 0.0, 0.0])
        #roughness = 0.0
        #specular = ti.Vector([0.0, 0.0, 0.0])
        g = 0.0
        gradient = ti.Vector([0.0, 0.0, 0.0])
        gradient_magnitude = 0.0

        while t < max_t:
            delta_t = -ti.log(1.0 - ti.random()) / sigma_max
            t += delta_t

            x = ray_origin + t * ray_dir
            voxel_idx = ti.floor((x - volume.volume_origin[None]) / volume.volume_spacing[None] + 0.5)
            
            if volume.inside_volume(voxel_idx):
                inside_volume = True

                density = sample_volume(volume, voxel_idx)
                tf_index = int(ti.math.clamp(density * (N_TF - 1), 0, N_TF - 1))
                opacity = tf_opacity[tf_index]

                p_accept = opacity / sigma_max
                if ti.random() < p_accept:
                    hit = True

                    color = tf_color[tf_index]
                    emission = tf_emission[tf_index]
                    #roughness = 0.5 #tf_roughness[tf_index]
                    #specular = ti.Vector([0.5, 0.5, 0.5]) #tf_specular[tf_index]
                    g = tf_phase_g[tf_index]

                    gradient = compute_gradient(volume, voxel_idx) / volume.volume_spacing[None]
                    gradient_magnitude = gradient.norm()
                    break

            elif inside_volume:
                break

        if not hit:
            radiance += throughput * ambient_light
            break

        gradient_factor = 0.5 
        p_surface = opacity * (1.0 - ti.exp(-25.0 * gradient_factor**3 * gradient_magnitude))

        is_surface = ti.random() < p_surface

        light_pos, light_dir, light_dist, cos_theta, pdf = light.sample_area_light(x)

        if cos_theta > 0.0:
            shadow_ray = Ray(x + 1e-3 * light_dir, light_dir)
            visible = trace_shadow_ray(shadow_ray, volume, light_dist)

            if visible:
                Le = light.color[None] * light.intensity[None]
                G = cos_theta / max(light_dist * light_dist, 1e-4)
                Ld = ti.Vector([0.0, 0.0, 0.0])

                if is_surface:
                    normal = gradient.normalized()
                    # view_dir = -ray_dir.normalized()
                    brdf = color / ti.math.pi # lambertian BRDF
                    Ld = throughput * brdf * Le * G / max(pdf, 1e-4) # * G
                else:
                    # phase = 1.0 / (4.0 * ti.math.pi) # Isotropic phase
                    phase = henyey_greenstein_phase(cos_theta, g)
                    Ld = throughput * color * phase * Le * G / max(pdf, 1e-4)

                emmission_contribution = throughput * emission
                radiance += Ld + emmission_contribution

        ambient_contribution = throughput * ambient_light * color
        radiance += ambient_contribution

        if is_surface:
            normal = gradient.normalized()
            ray_dir = sample_cosine_hemisphere(normal)
        else:
            ray_dir = sample_phase_function(ray_dir, g)

        ray_origin = x + 1e-4 * ray_dir 

        throughput *= color #* (1.0 - opacity)

        # russian roulette 
        if bounce > 2:
            p_terminate = 1.0 - ti.min(1.0, throughput.max())
            if ti.random() < p_terminate:
                throughput /= (1.0 - p_terminate)
                break

    return radiance

@ti.func
def sample_cosine_hemisphere(normal):
    u1 = ti.random()
    u2 = ti.random()
    r = ti.sqrt(u1)
    theta = 2.0 * ti.math.pi * u2

    x = r * ti.cos(theta)
    y = r * ti.sin(theta)
    z = ti.sqrt(1.0 - u1)

    w = normal.normalized()
    u = ti.Vector([0.0, 0.0, 0.0])
    if abs(w[0]) > 0.1:
        u = ti.Vector([0.0, 1.0, 0.0]).cross(w).normalized()
    else:
        u = ti.Vector([1.0, 0.0, 0.0]).cross(w).normalized()
    v = w.cross(u)

    return (u * x + v * y + w * z).normalized()

@ti.func
def sample_isotropic():
    z = 2.0 * ti.random() - 1.0
    r = ti.sqrt(max(0.0, 1.0 - z * z))
    phi = 2.0 * ti.math.pi * ti.random()
    x = r * ti.cos(phi)
    y = r * ti.sin(phi)
    return ti.Vector([x, y, z])

@ti.func
def sample_volume(volume: ti.template(), pos):
    x = ti.math.clamp(int(pos[0]), 0, volume.shape[0] - 1)
    y = ti.math.clamp(int(pos[1]), 0, volume.shape[1] - 1)
    z = ti.math.clamp(int(pos[2]), 0, volume.shape[2] - 1)
    
    return volume.volume[x, y, z]

@ti.func
def compute_gradient(volume: ti.template(), pos):
    eps_x = ti.Vector([1.0, 0.0, 0.0])
    eps_y = ti.Vector([0.0, 1.0, 0.0])
    eps_z = ti.Vector([0.0, 0.0, 1.0])
    
    dx = sample_volume(volume, pos + eps_x) - sample_volume(volume, pos - eps_x)
    dy = sample_volume(volume, pos + eps_y) - sample_volume(volume, pos - eps_y)
    dz = sample_volume(volume, pos + eps_z) - sample_volume(volume, pos - eps_z)
    
    return ti.Vector([dx, dy, dz]) * 0.5

@ti.func
def trace_shadow_ray(ray: ti.template(), volume: ti.template(), max_dist):
    t = 0.0
    visible = 1  

    while t < max_dist:
        u = ti.random()
        delta_t = -ti.log(1.0 - u)
        t += delta_t
        pos = ray.origin + ray.direction * t
        
        voxel_idx = ti.floor((pos - volume.volume_origin[None]) / volume.volume_spacing[None] + 0.5)
        
        if not volume.inside_volume(voxel_idx):
            break
            
        opacity = sample_volume(volume, voxel_idx)
        
        if ti.random() < opacity:
            visible = 0
            break
    
    return visible

@ti.func
def sample_sphere():
    z = 1.0 - 2.0 * ti.random()
    r = ti.sqrt(1.0 - z * z)
    phi = 2 * ti.math.pi * ti.random()
    x = r * ti.cos(phi)
    y = r * ti.sin(phi)
    return ti.Vector([x, y, z]).normalized()

@ti.func
def schlick_fresnel(F0, cos_theta):
    return F0 + (ti.Vector([1.0, 1.0, 1.0]) - F0) * ((1.0 - cos_theta) ** 5)

@ti.func
def fresnel_blend_brdf(normal, light_dir, view_dir, diffuse_color, specular_color, roughness):
    half_vec = (light_dir + view_dir).normalized()

    NdotL = max(0.0, normal.dot(light_dir))
    NdotV = max(0.0, normal.dot(view_dir))
    NdotH = max(0.0, normal.dot(half_vec))
    VdotH = max(0.0, view_dir.dot(half_vec))

    result = ti.Vector([0.0, 0.0, 0.0])
    
    valid = NdotL > 0.0 and NdotV > 0.0
    
    if valid:
        m = max(0.001, 2.0 / (roughness * roughness) - 2.0)
        D = ((m + 2.0) / (2.0 * ti.math.pi)) * (NdotH ** m)
        F = schlick_fresnel(specular_color, VdotH)
        G = NdotL * NdotV
        spec = (D * F) / max(4.0 * G, 1e-4)
        kd = ti.Vector([
            max(0.0, 1.0 - specular_color[0]),
            max(0.0, 1.0 - specular_color[1]),
            max(0.0, 1.0 - specular_color[2]),
        ])
        diff = kd * diffuse_color / ti.math.pi

        result = diff + spec
    
    return result
    

@ti.func
def henyey_greenstein_phase(cos_theta, g):
    g2 = g * g
    denom = 1.0 + g2 - 2.0 * g * cos_theta
    return (1.0 - g2) / (4.0 * ti.math.pi * denom * ti.sqrt(denom))

@ti.func
def sample_henyey_greenstein(g):
    u1 = ti.random()
    u2 = ti.random()

    cos_theta = 0.0
    if abs(g) < 1e-3:
        cos_theta = 1.0 - 2.0 * u1
    else:
        sqr_term = (1.0 - g * g) / (1.0 - g + 2.0 * g * u1)
        cos_theta = (1.0 + g * g - sqr_term * sqr_term) / (2.0 * g)

    sin_theta = ti.sqrt(1.0 - cos_theta * cos_theta)
    phi = 2.0 * ti.math.pi * u2
    x = sin_theta * ti.cos(phi)
    y = sin_theta * ti.sin(phi)
    z = cos_theta

    return ti.Vector([x, y, z]).normalized()

@ti.func
def sample_phase_function(ray_dir, g):
    local_dir = sample_henyey_greenstein(g)
    w = ray_dir.normalized()
    u = ti.Vector([0.0, 0.0, 0.0])
    if abs(w[0]) > 0.1:
        u = ti.Vector([0.0, 1.0, 0.0]).cross(w).normalized()
    else:
        u = ti.Vector([1.0, 0.0, 0.0]).cross(w).normalized()
    v = w.cross(u)
    
    world_dir = (u * local_dir[0] + v * local_dir[1] + w * local_dir[2]).normalized()
    return world_dir
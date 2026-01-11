import sys
import math
import numpy as np #as np so I don't have to type numpy everytime

#Dictionary to label indexes in object and light attribute arrays
IDX = {
    'NAME': 0, 'R': 1, 'G': 2, 'B': 3, 'KA': 4, 'KD': 5,
    'KS': 6, 'KR': 7, 'N': 8,  # Object shading properties
    'L_R': 1, 'L_G': 2, 'L_B': 3  # Light color components
}

#Function to Save image to a .ppm file
def save_ppm(filename, width, height, bg_color, pixels):
    with open(filename, 'w') as f:
        f.write("P3\n")
        f.write(f"{width} {height}\n255\n")
        for i in range(0, len(pixels), 3):
            color = pixels[i:i+3]
            for j in range(3):
                #Convert float [0,1] to int [0,255] and clamp values
                color[j] = int(min(max(color[j] * 255, 0), 255))
            f.write(f"{color[0]} {color[1]} {color[2]}\n")

#Normalize a vector to unit length
def unit_vector(v):
    return v / np.linalg.norm(v)

#Compute normalized direction vector from hit point to light
def direction_to_light(hit_pos, light_pos):
    return unit_vector(light_pos - hit_pos[:3])

#Calculate surface normal for a sphere under scaling
def surface_normal(pos, center, model_matrix):
    norm = np.array([
        (pos[0] - center[0]) / (model_matrix[0][0])**2,
        (pos[1] - center[1]) / (model_matrix[1][1])**2,
        (pos[2] - center[2]) / (model_matrix[2][2])**2
    ])
    return unit_vector(norm)

#Check whether a point is in shadow with respect to a light
def is_shadowed(point, light_dir, current_model, all_models, reverse_norm):
    for model in all_models:
        if (not np.array_equal(model, current_model)) or reverse_norm:
            inv_model = np.linalg.inv(model)
            local_origin = np.matmul(inv_model, point)[:3]
            ray_dir = np.matmul(inv_model, np.append(light_dir, 0))[:3]
            a = np.dot(ray_dir, ray_dir)
            b = 2 * np.dot(ray_dir, local_origin)
            c = np.dot(local_origin, local_origin) - 1
            disc = b**2 - 4*a*c
            if disc >= 0:
                t1 = (-b + math.sqrt(disc)) / (2*a)
                t2 = (-b - math.sqrt(disc)) / (2*a)
                if min(t1, t2) > 1e-6:
                    return True
    return False

#Compute Phong shading (diffuse + specular) at a surface point
def shade(hit_pos, normal, light_dir, obj_attr, light_attr, output, idx, shadow, view_dir):
    if shadow:
        return  # Skip if point is in shadow

    #Diffuse shading
    dot_nl = max(0, np.dot(normal, light_dir))
    if obj_attr[IDX['KD']] > 0:
        for c in range(3):
            output[idx + c] += obj_attr[IDX['KD']] * light_attr[c+1] * dot_nl * obj_attr[IDX['R']+c]

    #Specular shading
    if obj_attr[IDX['KS']] > 0 and dot_nl > 0:
        reflection = unit_vector(-light_dir - 2 * np.dot(-light_dir, normal) * normal)
        spec = max(0, np.dot(reflection, view_dir)) ** obj_attr[IDX['N']]
        for c in range(3):
            output[idx + c] += obj_attr[IDX['KS']] * light_attr[c+1] * spec

#Render the entire image from the scene data given in the input file
def render_scene(params, spheres, sphere_props, lights, light_props):
    width, height = params['res']
    bg = params['bg']
    image = bg * (width * height)  # Fill image with background color
    intersections = [float('inf')] * width * height  # Track closest hits
    eye = np.array([0, 0, 0, 1])  # Camera position at origin
    z_near = params['near']

    #For each sphere in the scene
    for s_idx, model in enumerate(spheres):
        inv_model = np.linalg.inv(model)
        origin = np.matmul(inv_model, eye)[:3]
        sphere_center = model[:, 3]
        obj_attr = sphere_props[s_idx]
        a_k, d_k, s_k, shininess = obj_attr[IDX['KA']], obj_attr[IDX['KD']], obj_attr[IDX['KS']], obj_attr[IDX['N']]

        #For each pixel in the image
        for y in range(height):
            for x in range(width):
                # Compute direction of ray for this pixel
                px = params['right'] * ((2 * x) / width - 1)
                py = -params['top'] * ((2 * y) / height - 1)
                direction = np.array([px, py, -z_near, 0])
                local_ray = np.matmul(inv_model, direction)[:3]

                #Ray-sphere intersection in local coordinates
                a = np.dot(local_ray, local_ray)
                b = 2 * np.dot(local_ray, origin)
                c = np.dot(origin, origin) - 1
                disc = b**2 - 4*a*c
                idx = (y * width + x) * 3

                if disc < 0:
                    continue  # No hit

                #Get closest valid intersection t
                t1 = (-b - math.sqrt(disc)) / (2*a)
                t2 = (-b + math.sqrt(disc)) / (2*a)
                t = None
                if t1 > z_near and t2 > z_near:
                    t = min(t1, t2)
                elif t1 > z_near or t2 > z_near:
                    t = max(t1, t2)
                if t is None or t >= intersections[y * width + x]:
                    continue

                intersections[y * width + x] = t

                #Transform to world coordinates
                global_ray = np.array([px, py, -z_near, 0])
                hit_world = eye + t * global_ray
                normal = surface_normal(hit_world, sphere_center, model)

                #Handle ray exiting the sphere (flip normal)
                flip = False
                if t1 < z_near and t2 > z_near:
                    normal = -normal
                    flip = True

                #Ambient component
                for i in range(3):
                    image[idx + i] = a_k * params['ambient'][i] * obj_attr[IDX['R'] + i]

                #Lighting and shading
                view_dir = unit_vector(-hit_world[:3])
                for l_idx, light_pos in enumerate(lights):
                    l_dir = direction_to_light(hit_world, light_pos)
                    in_shadow = is_shadowed(hit_world, l_dir, model, spheres, flip)
                    shade(hit_world, normal, l_dir, obj_attr, light_props[l_idx], image, idx, in_shadow, view_dir)

    #Save final image as a ppm file using the function
    save_ppm(params['output'], width, height, bg, np.clip(image, 0, 1))

#Parse the input file and initialize the scene
def parse_scene(file_path):
    params = {}
    spheres = []
    sphere_attrs = []
    lights = []
    light_attrs = []

    with open(file_path, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            if not tokens:
                continue
            key = tokens[0]
            if key == 'NEAR':
                params['near'] = float(tokens[1])
            elif key == 'RIGHT':
                params['right'] = float(tokens[1])
            elif key == 'TOP':
                params['top'] = float(tokens[1])
            elif key == 'RES':
                params['res'] = (int(tokens[1]), int(tokens[2]))
            elif key == 'SPHERE':
                #Create scaling + translation matrix for sphere
                matrix = np.array([
                    [float(tokens[5]), 0, 0, float(tokens[2])],
                    [0, float(tokens[6]), 0, float(tokens[3])],
                    [0, 0, float(tokens[7]), float(tokens[4])],
                    [0, 0, 0, 1]
                ])
                spheres.append(matrix)
                #Parse material attributes
                sphere_attrs.append([
                    tokens[1], float(tokens[8]), float(tokens[9]), float(tokens[10]),
                    float(tokens[11]), float(tokens[12]), float(tokens[13]),
                    float(tokens[14]), int(tokens[15])
                ])
            elif key == 'LIGHT':
                lights.append(np.array([float(tokens[2]), float(tokens[3]), float(tokens[4])]))
                light_attrs.append([
                    tokens[1], float(tokens[5]), float(tokens[6]), float(tokens[7])
                ])
            elif key == 'BACK':
                params['bg'] = [float(tokens[1]), float(tokens[2]), float(tokens[3])]
            elif key == 'AMBIENT':
                params['ambient'] = [float(tokens[1]), float(tokens[2]), float(tokens[3])]
            elif key == 'OUTPUT':
                params['output'] = tokens[1]

    render_scene(params, spheres, sphere_attrs, lights, light_attrs)

#Main funtion :)
if __name__ == "__main__":
    parse_scene(sys.argv[1])

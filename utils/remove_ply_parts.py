import numpy as np
from plyfile import PlyData, PlyElement
import random
import open3d as o3d


# Load the PLY file
plydata = PlyData.read('scans/scene0000_00/scene0000_00_vh_clean_2.ply')
point_cloud = o3d.io.read_point_cloud('scans/scene0000_00/scene0000_00_vh_clean_2.ply')

# Extract vertex data as numpy array
# vertices = np.array(plydata['vertex'].data)
# faces = np.array(plydata['face'].data)

# Define the dimensions of the area to exclude (in this case, 3m x 3m)
min_x = 0
max_x = 3
min_y = 0
max_y = 3
region_size = 4  # Minimum size in meters
bbox = point_cloud.get_oriented_bounding_box()
bbox_center = bbox.get_center()

# Step 4: Generate a random offset within the bounding box
random_offset = np.random.uniform(-region_size / 2, region_size / 2, size=(3,))
random_offset += bbox_center

# Step 5: Define the region to be removed (3m x 3m box centered at random_offset)
min_bound = random_offset - [region_size / 2, region_size / 2, region_size / 2]
max_bound = random_offset + [region_size / 2, region_size / 2, region_size / 2]

# Step 6: Filter out points within the specified region
filtered_points = []
filtered_colors = []


for point, color in zip(np.asarray(point_cloud.points), np.asarray(point_cloud.colors)):
    if not (min_bound[0] <= point[0] <= max_bound[0] and
            min_bound[1] <= point[1] <= max_bound[1] and
            min_bound[2] <= point[2] <= max_bound[2]):
        filtered_points.append(point)
        filtered_colors.append(color*255)
# print(filtered_colors)
# Step 4: Create a new PointCloud object with filtered points
print(filtered_points[0])
print(filtered_colors[0])

filtered_point_cloud = o3d.geometry.PointCloud()
# filtered_point_cloud.points = o3d.utility.Vector3dVector(filtered_points)
filtered_point_cloud.points = o3d.utility.Vector3dVector(point_cloud.points)
filtered_point_cloud.colors = o3d.utility.Vector3dVector(point_cloud.colors)

# Step 5: Save the filtered point cloud to a new PLY file
output_ply_path = 'output_filtered.ply'
o3d.io.write_point_cloud(output_ply_path, filtered_point_cloud)
point_cloud = o3d.io.read_point_cloud(output_ply_path)
print(len(np.asarray(point_cloud.colors)))

print(np.asarray(point_cloud.colors)[0])
exit(1)
filtered_vertices = []
filtered_material_indices = []
for vertex in vertices:
    x, y, z = vertex['x'], vertex['y'], vertex['z']
    if not (min_x <= x <= max_x and min_y <= y <= max_y):
        filtered_vertices.append(vertex)
        # filtered_vertices.append((vertex['x'],vertex['y'],vertex['z']))
        filtered_material_indices.append(faces)
print(filtered_material_indices)

# Create a new PlyElement with the filtered vertices
# filtered_vertex_element.data['material_index'] = filtered_material_indices
# filtered_plydata = PlyData([filtered_vertex_element] + [element for element in plydata.elements if element.name != 'vertex'])

# print(filtered_vertices)
# vertex_data = list(zip(x, y, z, red, green, blue))
filtered_vertex_element = PlyElement.describe(np.array(filtered_vertices), 'vertex', comments=['vertices'])
filtered_face_element = PlyElement.describe(np.array(filtered_material_indices), 'face', comments=['face'])

# Write the filtered element to a new PLY file
PlyData([filtered_face_element, filtered_vertex_element],text=False, byte_order='<').write('output_file.ply')
# filtered_plydata.write('output_file.ply')

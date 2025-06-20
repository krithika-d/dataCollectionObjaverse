import bpy
import mathutils
import sys
import math
import os

# Blender 4.1

# Parse command line arguments
argv = sys.argv
argv = argv[argv.index("--") + 1:]  # Get all args after "--"

if len(argv) != 18:
    print("Usage: blender -b -P blender_shadow.py -- <l> <r> <t> <path_mesh> <path_out>")
    sys.exit(1)
    
frame_id = argv[0]
lx, ly, lz = [float(x) for x in argv[1:1+3]]
r00, r01, r02, r10, r11, r12, r20, r21, r22 = [float(x) for x in argv[4:4+9]]
t0, t1, t2 = [float(x) for x in argv[13:13+3]]
obj_file_path = argv[16]
root_out = argv[17]

# Calibration parameters
camera_angle_x = 0.37403890205573304 # You may not need this if you use focal lengths
camera_angle_y = 0.6889539603385566  # You may not need this if you use focal lengths
fl_x = 4280.4869098293075 / 2  # Focal length in pixels
fl_y = 4281.150209131393 / 2  # Focal length in pixels
cx = 809.5  # Principal point x-coordinate (in pixels)
cy = 1535.5  # Principal point y-coordinate (in pixels)
w = 1620  # Image width in pixels
h = 3072  # Image height in pixels
k1 = 0.0  # Radial distortion coefficient
k2 = 0.0  # Radial distortion coefficient
p1 = 0.0  # Tangential distortion coefficient
p2 = 0.0  # Tangential distortion coefficient

# 4x4 transformation matrix from meshroom
transform_matrix = [
    [ r00, r01, r02, t0 ],
    [ r10, r11, r12, t1 ],
    [ r20, r21, r22, t2 ],
    [ 0.0, 0.0, 0.0, 1.0 ]
]

# Path to the OBJ file
# obj_file_path = "/dlbimg/volumetric/cache/ID_01245_XercesBlue_3K_r2020-Linear_G24int_Full_062023/00000001/Meshing/13149d66bfc766196f7a7cbf1ef78612a7e89af0/mesh.obj"

# Clear existing objects
bpy.ops.wm.read_factory_settings(use_empty=True)

# Create a cube
# bpy.ops.mesh.primitive_cube_add(size=2)
##bpy.ops.wm.obj_import(filepath=obj_file_path)
bpy.ops.import_scene.gltf(filepath=obj_file_path)

####
# cube = bpy.context.active_object
# cube.location = (0, 0, 0)

# # shade the cube smooth
# with bpy.context.temp_override(selected_editable_objects=[bpy.data.objects[cube.name]]):
#     bpy.ops.object.shade_smooth()
###
# Apply smooth shading to all imported mesh objects
for obj in bpy.context.scene.objects:
    if obj.type == 'MESH':
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.shade_smooth()
        obj.select_set(False)


# Create a point light
bpy.ops.object.light_add(type='POINT', radius=1, location=(lx, ly, lz))
light = bpy.context.active_object.data
light.energy = 1000

# Set the camera
bpy.ops.object.camera_add(location=(7, -7, 7))
camera = bpy.context.active_object
# Set camera intrinsic parameters
camera.data.lens = fl_x  # Focal length in mm (Blender uses mm)
camera.data.sensor_width = w / fl_x * camera.data.lens  # Sensor width in mm
camera.data.sensor_height = h / fl_y * camera.data.lens  # Sensor height in mm
# Set principal point offset
camera.data.shift_x = (cx - w / 2) / (w / 2)
camera.data.shift_y = (cy - h / 2) / (h / 2)
# Apply the transformation matrix to the camera
transform_matrix = mathutils.Matrix(transform_matrix)
blender_matrix = mathutils.Matrix(
    ((1, 0, 0, 0),
     (0, 0, 1, 0),
     (0, -1, 0, 0),
     (0, 0, 0, 1))
)
transform_matrix = blender_matrix @ transform_matrix

# Apply a 180-degree rotation around the Z-axis to correct the flip
rotation_180_y = mathutils.Matrix.Rotation(math.pi, 4, 'Y')
transform_matrix = transform_matrix @ rotation_180_y
rotation_180_z = mathutils.Matrix.Rotation(math.pi, 4, 'Z')
transform_matrix = transform_matrix @ rotation_180_z
camera.matrix_world = transform_matrix

# Set the scene camera
bpy.context.scene.camera = camera

############################ COMPOSITING ############################

# 
# root_out = '/dlbimg/projects/relightable_volumetric/logs/cycle'

# Enable the use of nodes for the compositor
bpy.context.scene.use_nodes = True

# Enable the depth pass
bpy.context.scene.view_layers['ViewLayer'].use_pass_z = True

# Clear existing nodes
bpy.context.scene.node_tree.nodes.clear()

# Create Render Layers node
render_layers = bpy.context.scene.node_tree.nodes.new(type='CompositorNodeRLayers')

# Create a Normalize node to normalize depth values
normalize_node = bpy.context.scene.node_tree.nodes.new(type='CompositorNodeNormalize')

# Create an Output File node for RGB image
rgb_output_node = bpy.context.scene.node_tree.nodes.new(type='CompositorNodeOutputFile')
rgb_output_node.label = 'RGB Output'
rgb_output_node.base_path = os.path.join(root_out, f'{lx}_{ly}_{lz}/render_{frame_id}')  # Change this to your desired output path
rgb_output_node.file_slots[0].path = 'rgb_'
rgb_output_node.format.file_format = 'PNG'

# Create an Output File node for Depth map
depth_output = bpy.context.scene.node_tree.nodes.new(type='CompositorNodeOutputFile')
depth_output.label = 'Depth Output'
depth_output.base_path = os.path.join(root_out, f'{lx}_{ly}_{lz}/render_{frame_id}')  # Change this to your desired output path
depth_output.file_slots[0].path = 'depth_'
depth_output.format.file_format = 'OPEN_EXR'

# Create an Output File node for normalized Depth map
normalized_depth_output = bpy.context.scene.node_tree.nodes.new(type='CompositorNodeOutputFile')
normalized_depth_output.label = 'Normalized Depth Output'
normalized_depth_output.base_path = os.path.join(root_out, f'{lx}_{ly}_{lz}/render_{frame_id}')  # Change this to your desired output path
normalized_depth_output.file_slots[0].path = 'depth_normalized_'
normalized_depth_output.format.file_format = 'PNG'
normalized_depth_output.format.color_mode = 'BW'

# Connect the nodes
bpy.context.scene.node_tree.links.new(render_layers.outputs['Image'], rgb_output_node.inputs[0])
bpy.context.scene.node_tree.links.new(render_layers.outputs['Depth'], depth_output.inputs[0])
bpy.context.scene.node_tree.links.new(render_layers.outputs['Depth'], normalize_node.inputs[0])
bpy.context.scene.node_tree.links.new(normalize_node.outputs[0], normalized_depth_output.inputs[0])

############################ RENDERING SETTINGS ############################

# rendering engines set to syscle
bpy.data.scenes[0].render.engine = "CYCLES"

# Set the device_type
bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "METAL" # or "OPENCL"

# Set the device and feature set
bpy.context.scene.cycles.device = "GPU"

# get_devices() to let Blender detects GPU device
bpy.context.preferences.addons["cycles"].preferences.get_devices()
print(bpy.context.preferences.addons["cycles"].preferences.compute_device_type)
for d in bpy.context.preferences.addons["cycles"].preferences.devices:
    d["use"] = 1 # Using all devices, include GPU and CPU
    print(d["name"], d["use"])

# Set the maximum samples for rendering
bpy.context.scene.cycles.samples = 8

# Set the render resolution
bpy.context.scene.render.resolution_x = w
bpy.context.scene.render.resolution_y = h
bpy.context.scene.render.resolution_percentage = 100

# Render the scene
bpy.ops.render.render(write_still=True)
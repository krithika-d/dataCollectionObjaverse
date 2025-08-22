import os
import json
import subprocess
import sys
import csv
from pathlib import Path
from datetime import datetime
import imageio.v2 as imageio
import bpy
import math
import mathutils

# Try to import tqdm, and in case its not available use fallbackkk
# 4 grid -albedo normal 2 lighting ; specular, metallic; upload github and server;
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    # Fallback progress bar implementation
    class tqdm:
        def __init__(self, iterable=None, **kwargs):
            self.iterable = iterable
            self.desc = kwargs.get('desc', '')
            self.total = len(iterable) if iterable else kwargs.get('total', 0)
            self.current = 0
            
        def __iter__(self):
            if self.iterable:
                for item in self.iterable:
                    self.current += 1
                    if self.current % 10 == 0:  # Print every 10 items
                        print(f"{self.desc}: {self.current}/{self.total}")
                    yield item
            else:
                for i in range(self.total):
                    self.current += 1
                    if self.current % 10 == 0:  # Print every 10 items
                        print(f"{self.desc}: {self.current}/{self.total}")
                    yield i
                    
        def update(self, n=1):
            self.current += n
            if self.current % 10 == 0:  # Print every 10 items
                print(f"{self.desc}: {self.current}/{self.total}")
    
    TQDM_AVAILABLE = False
    print("Warning: tqdm not available, using simple progress reporting")



# Input and Output Paths
INPUT_DIR = r"C:\Users\kdharanikota\Desktop\final_pipeline\combined_pipeline\test_objects"
OUTPUT_ROOT = r"C:\Users\kdharanikota\Desktop\final_pipeline\main\output_6"
RESULTS_DIR = r"C:\Users\kdharanikota\Desktop\final_pipeline\main\results_6"

# Blender Paths
BLENDER_ROOT = r"C:\Users\kdharanikota\Desktop\final_pipeline\blender-3.6.23-windows-x64"
BLENDER_PATH = r"C:\Users\kdharanikota\Desktop\final_pipeline\blender-3.6.23-windows-x64\blender.exe"

# File Extensions to Process
FILE_EXTENSIONS = ['.blend', '.fbx', '.obj', '.glb', '.gltf']
# '.stl', '.dae', '.ply', '.abc', '.usd', '.usda', '.usdc'

# Environment Maps for Rendering
ENVIRONMENT_MAPS = ['city', 'night']

# Render Settings (Optimized for Speed)
RENDER_SETTINGS = {
    'frames': 60,  # Reduced from 120 to 60 for 2x speed
    'resolution_x': 1280,  # Reduced from 1920 to 1280
    'resolution_y': 720   # Reduced from 1080 to 720
}

# Camera and Framing Settings
CAMERA_SETTINGS = {
    'fov': 60,  # Field of view in degrees (wider for better object fitting)
    'margin_factor': 1.25,  # Margin around object (1.0 = tight fit, 1.2 = more space)
    'camera_angle': 75,  # Camera angle in degrees
    'auto_adjust_distance': True,  # Automatically adjust camera distance
    'target_object_size': 1.2  # Target size for object scaling (smaller for more space)
}

# Processing Settings
PROCESSING_SETTINGS = {
    'timeout': 300  # 5 minutes timeout
}

# Material Check Settings
MATERIAL_CHECK_SETTINGS = {
    'min_materials': 1
}

# Output Settings
OUTPUT_SETTINGS = {
    'create_mp4': True,
    'render_normal_maps': True,  # Enable normal maps for checking
    'render_albedo_maps': True   # Enable albedo maps
}

# Grid Composition Settings
GRID_SETTINGS = {
    'max_objects': 25,  # Maximum objects to process for grid
    'grid_resolution': (1920, 1080),  # Output resolution for grid
    'create_grids': True,  # Enable grid creation
    'create_concatenated': True,  # Enable concatenated videos
    'lighting_transition_duration': 2.0,  # Duration for each lighting environment (seconds)
    'create_lighting_transitions': True  # Enable lighting transition compositions
}

# BLENDER FUNCTIONS (Built-in)



def clear_scene():
    """Clear all objects from the scene"""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    
    # Also clear materials to prevent naming conflicts
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    
    # Clear meshes to prevent naming conflicts
    for mesh in bpy.data.meshes:
        bpy.data.meshes.remove(mesh, do_unlink=True)
    
    # Clear images to prevent naming conflicts
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)

def set_object_origins(objects):
    """Set the origin of all mesh objects to their center of mass"""
    for obj in objects:
        if obj.type == 'MESH':
            # Select only this object
            bpy.ops.object.select_all(action='DESELECT')
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
            
            # Set origin to center of mass (better for centering)
            bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS', center='MEDIAN')
            
            print(f"✅ Set origin to center of mass for: {obj.name}")

def get_bounding_box(objects):
    """Calculate the bounding box of all objects"""
    if not objects:
        return None
    
    # Initialize with first object
    min_x = min_y = min_z = float('inf')
    max_x = max_y = max_z = float('-inf')
    
    for obj in objects:
        if obj.type == 'MESH':
            # Get world space bounds
            for vertex in obj.data.vertices:
                world_vertex = obj.matrix_world @ vertex.co
                min_x = min(min_x, world_vertex.x)
                min_y = min(min_y, world_vertex.y)
                min_z = min(min_z, world_vertex.z)
                max_x = max(max_x, world_vertex.x)
                max_y = max(max_y, world_vertex.y)
                max_z = max(max_z, world_vertex.z)
    
    if min_x == float('inf'):
        return None
    
    return {
        'min': mathutils.Vector((min_x, min_y, min_z)),
        'max': mathutils.Vector((max_x, max_y, max_z)),
        'center': mathutils.Vector(((min_x + max_x) / 2, (min_y + max_y) / 2, (min_z + max_z) / 2)),
        'size': mathutils.Vector((max_x - min_x, max_y - min_y, max_z - min_z))
    }

def center_and_scale_objects(objects, target_size=2.0):
    """Center all objects based on their Blender object centers and scale them to fill more of the frame"""
    if not objects:
        return None
    
    # First, set proper origins for all objects
    print("Setting object origins...")
    set_object_origins(objects)
    
    # Calculate bounding box
    bbox = get_bounding_box(objects)
    if not bbox:
        return None
    
    # Calculate size
    size = bbox['size']
    
    # Calculate scale factor based on the largest dimension
    max_dimension = max(size.x, size.y, size.z)
    
    # Calculate scale factor with better bounds
    if max_dimension > 0:
        scale_factor = target_size / max_dimension
        # Improved scaling bounds for better frame utilization
        # Allow objects to be larger (up to 3.0) for better detail visibility
        # But prevent them from being too small (minimum 0.1)
        scale_factor = max(0.1, min(3.0, scale_factor))
        
        # For very small objects, scale them up more aggressively
        if max_dimension < 0.5:
            scale_factor = min(5.0, scale_factor * 1.5)
        
        # For very large objects, scale them down more conservatively
        elif max_dimension > 5.0:
            scale_factor = max(0.05, scale_factor * 0.8)
    else:
        scale_factor = 1.0
    
    # Apply transformations to all objects
    for obj in objects:
        if obj.type == 'MESH':
            # Scale uniformly first
            obj.scale *= scale_factor
            
            # Then center based on object's origin (not geometric center)
            # Move the object so its origin is at world center (0,0,0)
            obj.location = (0, 0, 0)
    
    print(f"Centered objects at origin (using Blender object centers)")
    print(f"Scaled by factor: {scale_factor}")
    print(f"Original size: {max_dimension}, Target size: {target_size}")
    print(f"Final object dimensions: {size * scale_factor}")
    
    return scale_factor

def calculate_optimal_camera_distance(bbox, camera_fov=None, margin_factor=None):
    """Calculate optimal camera distance to fit object in frame with margin"""
    if camera_fov is None:
        camera_fov = CAMERA_SETTINGS['fov']
    if margin_factor is None:
        margin_factor = CAMERA_SETTINGS['margin_factor']
    
    if not bbox:
        return 6.0, 2.5
    
    # Get object dimensions
    size = bbox['size']
    width, height, depth = size.x, size.y, size.z
    
    # Calculate field of view in radians
    fov_rad = math.radians(camera_fov)
    
    # Calculate the maximum dimension that needs to fit in frame
    # Consider both width and height of the object
    max_horizontal = max(width, depth)  # Max width or depth
    max_vertical = height  # Height of object
    
    # Calculate required distance for horizontal fit
    # tan(fov/2) = (object_size/2) / distance
    # distance = (object_size/2) / tan(fov/2)
    horizontal_distance = (max_horizontal / 2) / math.tan(fov_rad / 2)
    
    # Calculate required distance for vertical fit
    # Account for camera angle - we need to project the height
    camera_angle_rad = math.radians(CAMERA_SETTINGS['camera_angle'])
    projected_height = height * math.cos(camera_angle_rad)
    vertical_distance = (projected_height / 2) / math.tan(fov_rad / 2)
    
    # Use the larger distance to ensure object fits in both dimensions
    # Add extra safety margin for complex shapes
    base_distance = max(horizontal_distance, vertical_distance) * 1.2
    
    # For very tall or wide objects, increase the margin further
    aspect_ratio = max_vertical / max_horizontal if max_horizontal > 0 else 1
    if aspect_ratio > 2.0:  # Very tall objects
        base_distance *= 1.3
    elif aspect_ratio < 0.5:  # Very wide objects
        base_distance *= 1.3
    
    # Apply margin factor for some breathing room
    optimal_distance = base_distance * margin_factor
    
    # Calculate camera height based on angle
    camera_height = optimal_distance * 0.4  # 40% of distance for 75-degree angle
    
    return optimal_distance, camera_height

def setup_camera_and_lighting(target_object, hdr_path):
    """Setup camera and lighting for optimal rendering"""
    scene = bpy.context.scene
    
    # Create or get camera
    if 'Camera' in bpy.data.objects:
        camera = bpy.data.objects['Camera']
    else:
        bpy.ops.object.camera_add(location=(0, -10, 5))
        camera = bpy.context.object
        camera.name = 'Camera'
    
    scene.camera = camera
    
    # Calculate optimal camera distance using improved algorithm
    bbox = get_bounding_box([target_object])
    radius, height = calculate_optimal_camera_distance(bbox)
    
    # Create pivot for camera animation at world origin
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 0))
    pivot = bpy.context.object
    pivot.name = 'CameraPivot'
    
    # Set pivot origin to world center (0,0,0)
    pivot.location = (0, 0, 0)
    
    # Parent camera to pivot
    camera.parent = pivot
    camera.location = (0, -radius, height)
    camera.rotation_euler = (math.radians(CAMERA_SETTINGS['camera_angle']), 0, 0)
    
    # Adjust camera view
    camera.data.type = 'PERSP'
    camera.data.lens = 50
    camera.data.clip_start = 0.1
    camera.data.clip_end = 1000.0
    
    # Add track-to constraint to look at object center
    constraint = camera.constraints.new(type='TRACK_TO')
    constraint.target = target_object
    constraint.track_axis = 'TRACK_NEGATIVE_Z'
    constraint.up_axis = 'UP_Y'
    
    # Set the camera as active
    bpy.context.scene.camera = camera
    
    # Ensure camera is looking at the object center
    camera.location = (0, -radius, height)
    camera.rotation_euler = (math.radians(CAMERA_SETTINGS['camera_angle']), 0, 0)
    
    # Additional camera adjustments for better framing
    camera.data.lens = 50  # Standard lens for better object framing #35 is wider lens
    camera.data.clip_start = 0.01  # Closer clip start
    camera.data.clip_end = 1000.0
    
    # Ensure object is perfectly centered by checking its location
    if target_object.location.length > 0.01:  # If object is not at origin
        print(f"⚠️  Object {target_object.name} is not at origin: {target_object.location}")
        # Move object to origin if it's not already there
        target_object.location = (0, 0, 0)
        print(f"✅ Moved {target_object.name} to origin (0,0,0)")
    
    # Fine-tune camera distance to ensure object fits perfectly
    def adjust_camera_for_perfect_fit():
        """Adjust camera distance to ensure object fits perfectly in frame"""
        # Get all mesh objects for bounding box calculation
        mesh_objects = [obj for obj in bpy.data.objects if obj.type == 'MESH']
        if not mesh_objects:
            return
        
        # Calculate bounding box of all objects
        all_bbox = get_bounding_box(mesh_objects)
        if not all_bbox:
            return
        
        # Recalculate optimal distance with more generous margin to prevent cutoff
        new_radius, new_height = calculate_optimal_camera_distance(all_bbox, margin_factor=1.15)
        
        # Update camera position
        camera.location = (0, -new_radius, new_height)
        
        print(f"🔧 Fine-tuned camera distance: {new_radius:.2f} (height: {new_height:.2f})")
    
    # Apply fine-tuning
    adjust_camera_for_perfect_fit()
    
    # Additional safety check - ensure camera is far enough to prevent cutoff
    def ensure_safe_camera_distance():
        """Ensure camera is at a safe distance to prevent object cutoff"""
        # Get all mesh objects for final bounding box calculation
        mesh_objects = [obj for obj in bpy.data.objects if obj.type == 'MESH']
        if not mesh_objects:
            return
        
        # Calculate final bounding box
        final_bbox = get_bounding_box(mesh_objects)
        if not final_bbox:
            return
        
        # Calculate minimum safe distance
        size = final_bbox['size']
        max_dimension = max(size.x, size.y, size.z)
        
        # Use a more conservative approach for complex shapes
        min_safe_distance = max_dimension * 2.5  # More conservative than before
        
        # If current distance is too close, increase it
        current_distance = camera.location.length
        if current_distance < min_safe_distance:
            scale_factor = min_safe_distance / current_distance
            camera.location *= scale_factor
            print(f"🔧 Increased camera distance for safety: {current_distance:.2f} → {min_safe_distance:.2f}")
    
    # Apply safety check
    ensure_safe_camera_distance()
    
    # Animate camera rotation around the object
    pivot.animation_data_clear()
    pivot.rotation_euler = (0, 0, 0)
    pivot.keyframe_insert(data_path="rotation_euler", frame=1)
    
    pivot.rotation_euler = (0, 0, math.radians(360))
    pivot.keyframe_insert(data_path="rotation_euler", frame=RENDER_SETTINGS['frames'])
    
    # Set linear interpolation for smooth rotation
    if pivot.animation_data and pivot.animation_data.action:
        for fcurve in pivot.animation_data.action.fcurves:
            for keyframe in fcurve.keyframe_points:
                keyframe.interpolation = 'LINEAR'
    
    # Setup HDR environment lighting
    if hdr_path and os.path.exists(hdr_path):
        world = bpy.context.scene.world
        if not world:
            world = bpy.data.worlds.new("World")
            bpy.context.scene.world = world
        
        world.use_nodes = True
        world_nodes = world.node_tree.nodes
        world_links = world.node_tree.links
        
        # Clear existing nodes
        world_nodes.clear()
        
        # Add World Output node
        world_output = world_nodes.new('ShaderNodeOutputWorld')
        world_output.location = (300, 0)
        
        # Add environment texture
        env_tex = world_nodes.new('ShaderNodeTexEnvironment')
        env_tex.location = (-300, 0)
        env_tex.image = bpy.data.images.load(hdr_path)
        
        # Add background shader
        background = world_nodes.new('ShaderNodeBackground')
        background.location = (0, 0)
        background.inputs['Strength'].default_value = 1.0
        
        # Connect nodes
        world_links.new(env_tex.outputs['Color'], background.inputs['Color'])
        world_links.new(background.outputs['Background'], world_output.inputs['Surface'])
        
        print(f"Loaded HDR environment: {hdr_path}")
    else:
        # Create simple world lighting
        world = bpy.context.scene.world
        if not world:
            world = bpy.data.worlds.new("World")
            bpy.context.scene.world = world
        
        world.use_nodes = True
        world_nodes = world.node_tree.nodes
        world_links = world.node_tree.links
        
        # Clear existing nodes
        world_nodes.clear()
        
        # Add World Output node
        world_output = world_nodes.new('ShaderNodeOutputWorld')
        world_output.location = (300, 0)
        
        # Add background shader
        background = world_nodes.new('ShaderNodeBackground')
        background.location = (0, 0)
        background.inputs['Color'].default_value = (0.1, 0.1, 0.1, 1.0)
        background.inputs['Strength'].default_value = 1.0
        
        # Connect nodes
        world_links.new(background.outputs['Background'], world_output.inputs['Surface'])
        
        print("Created simple world lighting")
    
    print(f"Camera setup complete - Distance: {radius:.2f}, Height: {height:.2f}")
    print(f"Pivot location: {pivot.location}")
    print(f"Target object location: {target_object.location}")
    print(f"Object size: {bbox['size'] if bbox else 'unknown'}")

#def setup_render_settings(output_path, resolution_x, resolution_y):
def setup_render_settings(output_path, resolution_x, resolution_y, render_normal_maps=True):
    """Setup render settings for high quality output"""
    scene = bpy.context.scene
    
    # Basic render settings
    scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath = output_path
    scene.render.use_file_extension = True
    scene.render.resolution_x = resolution_x
    scene.render.resolution_y = resolution_y
    scene.render.film_transparent = True
    
    ## Use Cycles render engine
    #scene.render.engine = 'CYCLES'
    # Use Eevee render engine for faster rendering
    scene.render.engine = 'BLENDER_EEVEE'
    
    # # GPU rendering
    # scene.cycles.device = 'GPU'
    
    # # Enable GPU compute devices
    # try:
    #     bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
    # except:
    #     try:
    #         bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'OPTIX'
    #     except:
    #         print("GPU rendering not available, using CPU")
    #         scene.cycles.device = 'CPU'
    
    # # Quality settings
    # scene.cycles.samples = 32
    # scene.cycles.preview_samples = 16
    # scene.cycles.use_denoising = True
    # scene.cycles.denoiser = 'OPTIX'

    # Eevee quality settings (Optimized for Speed)
    scene.eevee.taa_render_samples = 16  # Reduced from 64 to 16 for 4x speed
    scene.eevee.use_soft_shadows = False  # Disabled for speed
    scene.eevee.use_ssr = False  # Disabled Screen Space Reflections for speed
    scene.eevee.use_ssr_refraction = False  # Disabled for speed
    scene.eevee.use_bloom = False  # Disabled bloom for speed
    scene.eevee.bloom_threshold = 1.0
    scene.eevee.bloom_intensity = 0.05
    scene.eevee.use_motion_blur = False
    
    # Motion blur (optional)
    scene.render.use_motion_blur = False
    #print(f"Render settings configured - Resolution: {resolution_x}x{resolution_y}, Samples: {scene.cycles.samples}")
    #############################
    #NORMAL MAP TRY#
    # NORMAL MAP RENDERING WITH DIFFUSE SHADER Try 2
    # Setup normal map rendering if requested
    if render_normal_maps:
        print("🔧 Setting up normal map rendering with diffuse shader...")
        
        # Enable render layers for normal maps
        scene.render.use_compositing = True
        scene.use_nodes = True
        
        # Enable normal pass in render layers
        scene.view_layers[0].use_pass_normal = True
        
        # Get the compositor node tree
        if not scene.node_tree:
            scene.node_tree = bpy.data.node_trees.new(type='CompositorNodeTree')
        
        tree = scene.node_tree
        tree.nodes.clear()
        
        # Add render layers node
        render_layers = tree.nodes.new('CompositorNodeRLayers')
        render_layers.location = (0, 0)
        
        # Add composite output node for regular render
        composite = tree.nodes.new('CompositorNodeComposite')
        composite.location = (400, 0)
        
        # Add normal map output node
        normal_output = tree.nodes.new('CompositorNodeOutputFile')
        normal_output.location = (400, -200)
        normal_output.base_path = output_path.replace('frame_', 'normal_')
        normal_output.format.file_format = 'PNG'
        
        # Connect regular output
        tree.links.new(render_layers.outputs['Image'], composite.inputs['Image'])
        
        # For normal maps, we'll use a separate render with diffuse shader
        # The normal pass will be processed separately
        
        print(f"✅ Normal map rendering setup complete")
        print(f"   Normal files will be saved as: {output_path.replace('frame_', 'normal_')}")

    ######################
    
    print(f"Render settings configured - Resolution: {resolution_x}x{resolution_y}, Engine: Eevee, Samples: {scene.eevee.taa_render_samples}")
    #print(f"Render settings configured - Resolution: {resolution_x}x{resolution_y}, Engine: Cycles, Samples: {scene.cycles.samples}")

def import_model(file_path):
    """Import 3D model based on file extension"""
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == '.blend':
        bpy.ops.wm.open_mainfile(filepath=file_path)
        return True
    elif file_extension == '.fbx':
        bpy.ops.import_scene.fbx(filepath=file_path)
        return True
    elif file_extension == '.obj':
        bpy.ops.import_scene.obj(filepath=file_path)
        return True
    elif file_extension in ['.glb', '.gltf']:
        bpy.ops.import_scene.gltf(filepath=file_path)
        return True
    elif file_extension == '.stl':
        bpy.ops.import_mesh.stl(filepath=file_path)
        return True
    elif file_extension == '.dae':
        bpy.ops.wm.collada_import(filepath=file_path)
        return True
    elif file_extension == '.ply':
        bpy.ops.import_mesh.ply(filepath=file_path)
        return True
    elif file_extension == '.abc':
        bpy.ops.wm.alembic_import(filepath=file_path)
        return True
    elif file_extension in ['.usd', '.usda', '.usdc']:
        bpy.ops.wm.usd_import(filepath=file_path)
        return True
    else:
        # Try common importers as fallback
        importers = [
            ('fbx', bpy.ops.import_scene.fbx),
            ('obj', bpy.ops.import_scene.obj),
            ('gltf', bpy.ops.import_scene.gltf),
            ('stl', bpy.ops.import_mesh.stl)
        ]
        
        for name, importer in importers:
            try:
                importer(filepath=file_path)
                print(f"Successfully imported using {name} importer")
                return True
            except:
                continue
        
        return False

def check_materials():
    """Check materials in the current scene and return material info"""
    material_info = {
        "num_materials": 0,
        "materials": [],
        "objects_with_materials": 0,
        "objects_without_materials": 0
    }
    
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            if len(obj.material_slots) > 0:
                material_info["objects_with_materials"] += 1
                for slot in obj.material_slots:
                    if slot.material:
                        material_info["num_materials"] += 1
                        if slot.material.name not in material_info["materials"]:
                            material_info["materials"].append(slot.material.name)
            else:
                material_info["objects_without_materials"] += 1
    
    return material_info

def calculate_grid_dimensions(num_videos):
    """Calculate optimal grid dimensions for given number of videos"""
    if num_videos <= 0:
        return (0, 0)
    
    # Find the most square-like grid
    sqrt_videos = int(math.sqrt(num_videos))
    
    # Try to make it as square as possible
    if sqrt_videos * sqrt_videos >= num_videos:
        # Perfect square or close to it
        cols = sqrt_videos
        rows = (num_videos + cols - 1) // cols  # Ceiling division
    else:
        # Need more columns
        cols = sqrt_videos + 1
        rows = (num_videos + cols - 1) // cols
    
    return (cols, rows)

def create_video_grid(video_paths, output_path, grid_width, grid_height, target_resolution=(1920, 1080)):
    """Create a grid video from multiple video files using FFMPEG"""
    if not video_paths:
        print("❌ No video paths provided for grid creation")
        return False
    
    # Calculate individual video size
    cell_width = target_resolution[0] // grid_width
    cell_height = target_resolution[1] // grid_height
    
    # Build FFMPEG command for grid composition
    input_videos = ""
    filter_complex = f"nullsrc=size={target_resolution[0]}x{target_resolution[1]} [base];"
    overlay_chain = ""
    
    for i, video_path in enumerate(video_paths):
        if i >= grid_width * grid_height:
            break  # Limit to grid size
        
        # Calculate position in grid
        col = i % grid_width
        row = i // grid_width
        x_pos = col * cell_width
        y_pos = row * cell_height
        
        # Add input
        input_videos += f" -i \"{video_path}\""
        
        # Add scaling and positioning
        filter_complex += f"[{i}:v] setpts=PTS-STARTPTS, scale={cell_width}x{cell_height} [video{i}];"
        
        # Add overlay
        if i == 0:
            overlay_chain += f"[base][video{i}] overlay=shortest=1:x={x_pos}:y={y_pos} [tmp{i}];"
        elif i < len(video_paths) - 1 and i < grid_width * grid_height - 1:
            overlay_chain += f"[tmp{i-1}][video{i}] overlay=shortest=1:x={x_pos}:y={y_pos} [tmp{i}];"
        else:
            overlay_chain += f"[tmp{i-1}][video{i}] overlay=shortest=1:x={x_pos}:y={y_pos}"
    
    # Complete FFMPEG command
    cmd = f'ffmpeg{input_videos} -filter_complex "{filter_complex}{overlay_chain}" -c:v libx264 -preset fast -crf 23 -y "{output_path}"'
    
    print(f"🎬 Creating grid video: {grid_width}x{grid_height}")
    print(f"📁 Output: {output_path}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Grid video created successfully: {output_path}")
            return True
        else:
            print(f"❌ FFMPEG failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error creating grid video: {e}")
        return False


def create_lighting_transition_grid(rendered_models, output_root, transition_duration=2.0):
    """Create grid composition with lighting transition (city -> night)"""
    print("\n🎬 Creating Lighting Transition Grid Compositions")
    print("=" * 60)
    
    # Group videos by object name
    object_videos = {}
    for model_info in rendered_models:
        file_name = os.path.splitext(os.path.basename(model_info['file']))[0]
        object_videos[file_name] = {}
        
        for envmap in model_info['rendered_envmaps']:
            # Check in model_videos directory first
            video_path = os.path.join(output_root, "model_videos", f"{file_name}_{envmap}.mp4")
            if os.path.exists(video_path):
                object_videos[file_name][envmap] = video_path
            else:
                # Fallback to old location
                video_path = os.path.join(output_root, f"{file_name}_{envmap}.mp4")
                if os.path.exists(video_path):
                    object_videos[file_name][envmap] = video_path
    
    # Filter objects that have both city and night videos
    complete_objects = {}
    partial_objects = {}
    
    for obj_name, envmaps in object_videos.items():
        if 'city' in envmaps and 'night' in envmaps:
            complete_objects[obj_name] = envmaps
        elif 'city' in envmaps or 'night' in envmaps:
            partial_objects[obj_name] = envmaps
    
    print(f"📊 Found {len(complete_objects)} objects with both city and night lighting")
    print(f"📊 Found {len(partial_objects)} objects with only one lighting environment")
    
    if len(complete_objects) == 0:
        print("❌ No objects found with both lighting environments")
        print("💡 Creating lighting transition with available objects...")
        # Use objects with at least one lighting environment
        if len(partial_objects) > 0:
            complete_objects = partial_objects
        else:
            return
    
    # Create compositions directory
    compositions_dir = os.path.join(output_root, "compositions")
    os.makedirs(compositions_dir, exist_ok=True)
    
    # Limit to max objects for grid
    max_objects = min(len(complete_objects), GRID_SETTINGS['max_objects'])
    selected_objects = list(complete_objects.items())[:max_objects]
    
    # Calculate grid dimensions
    cols, rows = calculate_grid_dimensions(len(selected_objects))
    print(f"📐 Grid dimensions: {cols}x{rows} for {len(selected_objects)} objects")
    
    # Create the lighting transition grid
    grid_output = os.path.join(compositions_dir, f"lighting_transition_{cols}x{rows}_{len(selected_objects)}objects.mp4")
    
    if create_lighting_transition_grid_video(selected_objects, grid_output, cols, rows, transition_duration):
        print(f"✅ Lighting transition grid created: {grid_output}")
    
    # Create smaller grids if we have enough objects
    if len(selected_objects) >= 16:
        # 4x4 grid
        selected_4x4 = selected_objects[:16]
        grid_4x4_output = os.path.join(compositions_dir, "lighting_transition_4x4_16objects.mp4")
        if create_lighting_transition_grid_video(selected_4x4, grid_4x4_output, 4, 4, transition_duration):
            print(f"✅ 4x4 lighting transition grid created: {grid_4x4_output}")
    
    if len(selected_objects) >= 9:
        # 3x3 grid
        selected_3x3 = selected_objects[:9]
        grid_3x3_output = os.path.join(compositions_dir, "lighting_transition_3x3_9objects.mp4")
        if create_lighting_transition_grid_video(selected_3x3, grid_3x3_output, 3, 3, transition_duration):
            print(f"✅ 3x3 lighting transition grid created: {grid_3x3_output}")
    
    print(f"\n🎬 Lighting transition compositions saved to: {compositions_dir}")

def create_lighting_transition_grid_video(selected_objects, output_path, grid_width, grid_height, transition_duration=2.0):
    """Create a grid video with lighting transition from city to night using simpler approach"""
    if not selected_objects:
        print("❌ No objects provided for grid creation")
        return False
    
    # Calculate individual video size
    target_resolution = GRID_SETTINGS['grid_resolution']
    cell_width = target_resolution[0] // grid_width
    cell_height = target_resolution[1] // grid_height
    
    # Limit to grid size
    max_objects = grid_width * grid_height
    selected_objects = selected_objects[:max_objects]
    
    print(f"🎬 Creating lighting transition grid: {grid_width}x{grid_height}")
    print(f"📁 Output: {output_path}")
    print(f"⏱️  Duration: {transition_duration * 2} seconds (2s city + 2s night)")
    
    # Create temporary files for city and night grids
    temp_city_grid = output_path.replace('.mp4', '_city_temp.mp4')
    temp_night_grid = output_path.replace('.mp4', '_night_temp.mp4')
    
    # Step 1: Create city grid (first 2 seconds)
    city_videos = []
    for i, (obj_name, envmaps) in enumerate(selected_objects):
        if i >= grid_width * grid_height:
            break
        
        # Get city video (or fallback to night if city not available)
        if 'city' in envmaps:
            city_video = envmaps['city']
        elif 'night' in envmaps:
            city_video = envmaps['night']  # Fallback
        else:
            continue  # Skip if no video available
        
        city_videos.append(city_video)
    
    if city_videos:
        # Create city grid using simple approach
        if create_simple_grid(city_videos, temp_city_grid, grid_width, grid_height, target_resolution, transition_duration):
            print(f"✅ Created city grid: {temp_city_grid}")
        else:
            print(f"❌ Failed to create city grid")
            return False
    
    # Step 2: Create night grid (second 2 seconds)
    night_videos = []
    for i, (obj_name, envmaps) in enumerate(selected_objects):
        if i >= grid_width * grid_height:
            break
        
        # Get night video (or fallback to city if night not available)
        if 'night' in envmaps:
            night_video = envmaps['night']
        elif 'city' in envmaps:
            night_video = envmaps['city']  # Fallback
        else:
            continue  # Skip if no video available
        
        night_videos.append(night_video)
    
    if night_videos:
        # Create night grid using simple approach
        if create_simple_grid(night_videos, temp_night_grid, grid_width, grid_height, target_resolution, transition_duration):
            print(f"✅ Created night grid: {temp_night_grid}")
        else:
            print(f"❌ Failed to create night grid")
            return False
    
    # Step 3: Concatenate the two grids
    try:
        cmd = f'ffmpeg -i "{temp_city_grid}" -i "{temp_night_grid}" -filter_complex "[0:v][1:v]concat=n=2:v=1:a=0" -c:v libx264 -preset fast -crf 23 -y "{output_path}"'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Lighting transition grid created successfully: {output_path}")
            # Clean up temp files
            try:
                os.remove(temp_city_grid)
                os.remove(temp_night_grid)
            except:
                pass
            return True
        else:
            print(f"❌ FFMPEG concatenation failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error creating lighting transition grid: {e}")
        return False

def create_simple_grid(video_paths, output_path, grid_width, grid_height, target_resolution, duration=2.0):
    """Create a simple grid from video paths"""
    if not video_paths:
        return False
    
    # Calculate individual video size
    cell_width = target_resolution[0] // grid_width
    cell_height = target_resolution[1] // grid_height
    
    # Build FFMPEG command for simple grid
    input_videos = ""
    filter_complex = f"nullsrc=size={target_resolution[0]}x{target_resolution[1]}:duration={duration} [base];"
    overlay_chain = ""
    
    for i, video_path in enumerate(video_paths):
        if i >= grid_width * grid_height or video_path is None:
            break
        
        # Calculate position in grid
        col = i % grid_width
        row = i // grid_width
        x_pos = col * cell_width
        y_pos = row * cell_height
        
        # Add input
        input_videos += f" -i \"{video_path}\""
        
        # Add scaling and positioning
        filter_complex += f"[{i}:v] setpts=PTS-STARTPTS, scale={cell_width}x{cell_height}, trim=duration={duration} [video{i}];"
        
        # Add overlay
        if i == 0:
            overlay_chain += f"[base][video{i}] overlay=shortest=1:x={x_pos}:y={y_pos} [tmp{i}];"
        elif i < len(video_paths) - 1 and i < grid_width * grid_height - 1:
            overlay_chain += f"[tmp{i-1}][video{i}] overlay=shortest=1:x={x_pos}:y={y_pos} [tmp{i}];"
        else:
            overlay_chain += f"[tmp{i-1}][video{i}] overlay=shortest=1:x={x_pos}:y={y_pos}"
    
    # Complete FFMPEG command
    cmd = f'ffmpeg{input_videos} -filter_complex "{filter_complex}{overlay_chain}" -c:v libx264 -preset fast -crf 23 -y "{output_path}"'
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error creating simple grid: {e}")
        return False

def create_video_compositions(rendered_models, output_root):
    """Create grid compositions with lighting transitions"""
    if not GRID_SETTINGS['create_grids'] and not GRID_SETTINGS['create_concatenated']:
        return
    
    print("\n🎬 Creating Video Compositions")
    print("=" * 50)
    
    # Create lighting transition compositions (new feature)
    if GRID_SETTINGS['create_lighting_transitions']:
        create_lighting_transition_grid(rendered_models, output_root, transition_duration=GRID_SETTINGS['lighting_transition_duration'])
    else:
        print("⏭️  Lighting transition compositions disabled")
    
    # Original grid composition (keeping for backward compatibility)
    # Collect all successful video paths
    all_videos = []
    for model_info in rendered_models:
        file_name = os.path.splitext(os.path.basename(model_info['file']))[0]
        for envmap in model_info['rendered_envmaps']:
            # Check in model_videos directory first
            video_path = os.path.join(output_root, "model_videos", f"{file_name}_{envmap}.mp4")
            if os.path.exists(video_path):
                all_videos.append({
                    'path': video_path,
                    'name': file_name,
                    'envmap': envmap
                })
            else:
                # Fallback to old location
                video_path = os.path.join(output_root, f"{file_name}_{envmap}.mp4")
                if os.path.exists(video_path):
                    all_videos.append({
                        'path': video_path,
                        'name': file_name,
                        'envmap': envmap
                    })
    
    print(f"📊 Found {len(all_videos)} videos for traditional composition")
    
    if len(all_videos) == 0:
        print("❌ No videos found for traditional composition")
        return
    
    # Create compositions directory
    compositions_dir = os.path.join(output_root, "compositions")
    os.makedirs(compositions_dir, exist_ok=True)
    
    # Create traditional grid videos (optional)
    if GRID_SETTINGS['create_grids']:
        print("\n🎬 Creating Traditional Grid Videos...")
        
        # Limit to max objects
        max_videos = min(len(all_videos), GRID_SETTINGS['max_objects'])
        selected_videos = all_videos[:max_videos]
        
        # Calculate grid dimensions
        cols, rows = calculate_grid_dimensions(len(selected_videos))
        print(f"📐 Grid dimensions: {cols}x{rows} for {len(selected_videos)} videos")
        
        # Create grid video
        video_paths = [v['path'] for v in selected_videos]
        grid_output = os.path.join(compositions_dir, f"traditional_grid_{cols}x{rows}_{len(selected_videos)}videos.mp4")
        
        if create_video_grid(video_paths, grid_output, cols, rows, GRID_SETTINGS['grid_resolution']):
            print(f"✅ Traditional grid video created: {grid_output}")
    
    print(f"\n🎬 All video compositions saved to: {compositions_dir}")
    
    # Create grid compositions for normal and albedo maps
    create_map_type_grids(rendered_models, output_root)

def create_map_type_grids(rendered_models, output_root):
    """Create grid compositions for normal and albedo maps"""
    print("\n🎬 Creating Map Type Grid Compositions")
    print("=" * 50)
    
    # Create compositions directory
    compositions_dir = os.path.join(output_root, "compositions")
    os.makedirs(compositions_dir, exist_ok=True)
    
    # Collect normal map videos
    normal_videos = []
    normal_videos_dir = os.path.join(output_root, "normal_map_videos")
    if os.path.exists(normal_videos_dir):
        for file in os.listdir(normal_videos_dir):
            if file.endswith('.mp4'):
                normal_videos.append({
                    'path': os.path.join(normal_videos_dir, file),
                    'name': file.replace('.mp4', ''),
                    'type': 'normal'
                })
    
    # Collect albedo map videos
    albedo_videos = []
    albedo_videos_dir = os.path.join(output_root, "albedo_videos")
    if os.path.exists(albedo_videos_dir):
        for file in os.listdir(albedo_videos_dir):
            if file.endswith('.mp4'):
                albedo_videos.append({
                    'path': os.path.join(albedo_videos_dir, file),
                    'name': file.replace('.mp4', ''),
                    'type': 'albedo'
                })
    
    print(f"📊 Found {len(normal_videos)} normal map videos")
    print(f"📊 Found {len(albedo_videos)} albedo map videos")
    
    # Create normal map grid
    if len(normal_videos) > 0:
        max_videos = min(len(normal_videos), GRID_SETTINGS['max_objects'])
        selected_normal_videos = normal_videos[:max_videos]
        
        cols, rows = calculate_grid_dimensions(len(selected_normal_videos))
        print(f"📐 Normal map grid dimensions: {cols}x{rows} for {len(selected_normal_videos)} videos")
        
        video_paths = [v['path'] for v in selected_normal_videos]
        normal_grid_output = os.path.join(compositions_dir, f"normal_maps_grid_{cols}x{rows}_{len(selected_normal_videos)}videos.mp4")
        
        if create_video_grid(video_paths, normal_grid_output, cols, rows, GRID_SETTINGS['grid_resolution']):
            print(f"✅ Normal maps grid video created: {normal_grid_output}")
    
    # Create albedo map grid
    if len(albedo_videos) > 0:
        max_videos = min(len(albedo_videos), GRID_SETTINGS['max_objects'])
        selected_albedo_videos = albedo_videos[:max_videos]
        
        cols, rows = calculate_grid_dimensions(len(selected_albedo_videos))
        print(f"📐 Albedo map grid dimensions: {cols}x{rows} for {len(selected_albedo_videos)} videos")
        
        video_paths = [v['path'] for v in selected_albedo_videos]
        albedo_grid_output = os.path.join(compositions_dir, f"albedo_maps_grid_{cols}x{rows}_{len(selected_albedo_videos)}videos.mp4")
        
        if create_video_grid(video_paths, albedo_grid_output, cols, rows, GRID_SETTINGS['grid_resolution']):
            print(f"✅ Albedo maps grid video created: {albedo_grid_output}")
    
    print(f"\n🎬 Map type grid compositions saved to: {compositions_dir}")

def render_normal_maps_with_eevee(mesh_objects, output_dir, envmap):
    """Render normal maps using Eevee with proper normal pass"""
    print(f"🎨 Rendering normal maps with Eevee for {envmap}...")
    
    scene = bpy.context.scene
    
    # Ensure we're using Eevee
    scene.render.engine = 'BLENDER_EEVEE'
    
    # Setup compositor for normal pass
    scene.render.use_compositing = True
    scene.use_nodes = True
    
    # Enable normal pass in render layers
    scene.view_layers[0].use_pass_normal = True
    
    # Get the compositor node tree
    if not scene.node_tree:
        scene.node_tree = bpy.data.node_trees.new(type='CompositorNodeTree')
    
    tree = scene.node_tree
    tree.nodes.clear()
    
    # Add render layers node
    render_layers = tree.nodes.new('CompositorNodeRLayers')
    render_layers.location = (0, 0)
    
    # Add normal map output node
    normal_output = tree.nodes.new('CompositorNodeOutputFile')
    normal_output.location = (400, 0)
    normal_output.base_path = os.path.join(output_dir, f"normal_{envmap}_")
    normal_output.format.file_format = 'PNG'
    
    # Connect normal pass to output
    tree.links.new(render_layers.outputs['Normal'], normal_output.inputs['Image'])
    
    # Set animation range
    scene.frame_start = 1
    scene.frame_end = RENDER_SETTINGS['frames']
    
    # Render normal maps with Eevee
    print(f"🎬 Rendering {RENDER_SETTINGS['frames']} normal map frames with Eevee...")
    bpy.ops.render.render(animation=True)
    
    print(f"✅ Eevee normal map rendering complete for {envmap}")
    print(f"   Normal maps saved as: {os.path.join(output_dir, f'normal_{envmap}_*.png')}")
    
    # Check if normal maps are actually normal maps (not diffuse)
    check_normal_map_quality(output_dir, envmap)

def render_albedo_maps_with_eevee(mesh_objects, output_dir, envmap):
    """Render albedo maps using Eevee with emission shaders for guaranteed color visibility"""
    print(f"🎨 Rendering albedo maps with Eevee for {envmap}...")
    
    scene = bpy.context.scene
    
    # Store original materials for restoration
    original_materials = {}
    for obj in mesh_objects:
        if obj.type == 'MESH':
            original_materials[obj] = []
            for slot in obj.material_slots:
                if slot.material:
                    original_materials[obj].append(slot.material)
    
    # Create emission-based albedo materials for guaranteed color visibility
    albedo_materials = {}
    for obj in mesh_objects:
        if obj.type == 'MESH':
            albedo_materials[obj] = []
            for i, slot in enumerate(obj.material_slots):
                if slot.material:
                    # Create a new material for albedo
                    albedo_mat = bpy.data.materials.new(name=f"albedo_{obj.name}_{i}")
                    albedo_mat.use_nodes = True
                    
                    # Clear existing nodes
                    albedo_mat.node_tree.nodes.clear()
                    
                    # Add emission shader for guaranteed color visibility
                    emission = albedo_mat.node_tree.nodes.new('ShaderNodeEmission')
                    emission.location = (0, 0)
                    
                    # Get color from original material
                    original_mat = slot.material
                    
                    # Extract base color and texture properly
                    base_color = (0.8, 0.8, 0.8, 1.0)  # Default gray
                    base_color_texture = None
                    
                    if original_mat.use_nodes:
                        # Look for base color texture or color in principled BSDF
                        for node in original_mat.node_tree.nodes:
                            if node.type == 'BSDF_PRINCIPLED' and 'Base Color' in node.inputs:
                                if node.inputs['Base Color'].links:
                                    # Get the texture node connected to base color
                                    base_color_texture = node.inputs['Base Color'].links[0].from_node
                                    base_color = (1.0, 1.0, 1.0, 1.0)  # White to show texture properly
                                else:
                                    # Use the original base color and make it brighter
                                    original_color = node.inputs['Base Color'].default_value
                                    base_color = tuple(min(1.0, c * 1.5) for c in original_color[:3]) + (1.0,)
                                break
                            elif node.type == 'TEX_IMAGE':  # Fallback: direct image texture
                                base_color_texture = node
                                base_color = (1.0, 1.0, 1.0, 1.0)
                                break
                    else:
                        # Use diffuse color and make it brighter
                        original_color = original_mat.diffuse_color
                        base_color = tuple(min(1.0, c * 1.5) for c in original_color[:3]) + (1.0,)
                    
                    # Set emission color and strength
                    emission.inputs['Color'].default_value = base_color
                    emission.inputs['Strength'].default_value = 2.0  # Bright emission
                    
                    # If we found a texture, properly copy it to the albedo material
                    if base_color_texture and base_color_texture.type == 'TEX_IMAGE':
                        # Copy the texture node to the albedo material
                        copied_texture = albedo_mat.node_tree.nodes.new('ShaderNodeTexImage')
                        copied_texture.location = (-200, 0)
                        
                        # Copy the image data
                        if base_color_texture.image:
                            copied_texture.image = base_color_texture.image
                            # Connect the copied texture to emission
                            albedo_mat.node_tree.links.new(copied_texture.outputs['Color'], emission.inputs['Color'])
                            print(f"  🎨 Copied and connected texture for {obj.name} material {i}: {base_color_texture.image.name}")
                        else:
                            # If no image data, use the color directly
                            emission.inputs['Color'].default_value = base_color
                            print(f"  ⚠️  Texture found but no image data for {obj.name} material {i}")
                    else:
                        print(f"  🎨 Using solid color for {obj.name} material {i}: {base_color[:3]}")
                    
                    # Add material output
                    material_output = albedo_mat.node_tree.nodes.new('ShaderNodeOutputMaterial')
                    material_output.location = (200, 0)
                    
                    # Connect emission to output
                    albedo_mat.node_tree.links.new(emission.outputs['Emission'], material_output.inputs['Surface'])
                    
                    # Assign the albedo material
                    slot.material = albedo_mat
                    albedo_materials[obj].append(albedo_mat)
                    
                    print(f"  🎨 Created albedo material for {obj.name} material {i}: {base_color[:3]}")
    
    # Setup pure black background to make emission colors pop
    world = bpy.context.scene.world
    if not world:
        world = bpy.data.worlds.new("AlbedoWorld")
        bpy.context.scene.world = world
    
    world.use_nodes = True
    world_nodes = world.node_tree.nodes
    world_links = world.node_tree.links
    
    # Clear existing nodes
    world_nodes.clear()
    
    # Add World Output node
    world_output = world_nodes.new('ShaderNodeOutputWorld')
    world_output.location = (300, 0)
    
    # Add pure black background to make emission colors pop
    background = world_nodes.new('ShaderNodeBackground')
    background.location = (0, 0)
    background.inputs['Color'].default_value = (0.0, 0.0, 0.0, 1.0)  # Pure black
    background.inputs['Strength'].default_value = 0.0  # No background lighting
    
    # Connect background to world output
    world_links.new(background.outputs['Background'], world_output.inputs['Surface'])
    
    # Setup render settings for albedo
    scene.render.use_compositing = True
    scene.use_nodes = True
    
    # Get the compositor node tree
    if not scene.node_tree:
        scene.node_tree = bpy.data.node_trees.new(type='CompositorNodeTree')
    
    tree = scene.node_tree
    tree.nodes.clear()
    
    # Add render layers node
    render_layers = tree.nodes.new('CompositorNodeRLayers')
    render_layers.location = (0, 0)
    
    # Add albedo map output node
    albedo_output = tree.nodes.new('CompositorNodeOutputFile')
    albedo_output.location = (400, 0)
    albedo_output.base_path = os.path.join(output_dir, f"albedo_{envmap}_")
    albedo_output.format.file_format = 'PNG'
    
    # Connect to regular image output
    tree.links.new(render_layers.outputs['Image'], albedo_output.inputs['Image'])
    
    # Set animation range
    scene.frame_start = 1
    scene.frame_end = RENDER_SETTINGS['frames']
    
    # Render albedo maps with Eevee
    print(f"🎬 Rendering {RENDER_SETTINGS['frames']} albedo map frames with Eevee...")
    bpy.ops.render.render(animation=True)
    
    print(f"✅ Eevee albedo map rendering complete for {envmap}")
    print(f"   Albedo maps saved as: {os.path.join(output_dir, f'albedo_{envmap}_*.png')}")
    
    # Restore original materials
    for obj in mesh_objects:
        if obj.type == 'MESH' and obj in original_materials:
            for i, slot in enumerate(obj.material_slots):
                if i < len(original_materials[obj]):
                    slot.material = original_materials[obj][i]
    
    # Clean up albedo materials
    for obj_materials in albedo_materials.values():
        for mat in obj_materials:
            bpy.data.materials.remove(mat, do_unlink=True)
    
    print("✅ Restored original materials")
    
    # Check albedo map quality
    check_albedo_map_quality(output_dir, envmap)

def check_normal_map_quality(output_dir, envmap):
    """Check if normal maps are actually colorful normal maps, not diffuse renders"""
    import imageio
    import numpy as np
    
    # Find the normal map subdirectory
    normal_subdir = os.path.join(output_dir, f"normal_{envmap}_")
    
    if not os.path.exists(normal_subdir):
        print(f"⚠️  No normal map directory found: {normal_subdir}")
        return
    
    # Find the first normal map file
    normal_files = [f for f in os.listdir(normal_subdir) if f.startswith("Image") and f.endswith('.png')]
    
    if not normal_files:
        print(f"⚠️  No normal map files found in {normal_subdir}")
        return
    
    # Check the first normal map file
    normal_file = os.path.join(normal_subdir, normal_files[0])
    print(f"🔍 Checking normal map quality: {normal_file}")
    
    try:
        # Load the image
        img = imageio.imread(normal_file)
        
        if len(img.shape) == 3 and img.shape[2] >= 3:  # RGB or RGBA
            # Check if it's colorful (has variation in all channels)
            r_channel = img[:, :, 0]
            g_channel = img[:, :, 1] 
            b_channel = img[:, :, 2]
            
            # Calculate statistics
            r_std = np.std(r_channel)
            g_std = np.std(g_channel)
            b_std = np.std(b_channel)
            
            # Check if all channels have variation (indicating normal data)
            has_variation = r_std > 10 and g_std > 10 and b_std > 10
            
            # Check if it's not just grayscale
            is_colorful = np.std([r_std, g_std, b_std]) > 5
            
            print(f"📊 Normal map analysis:")
            print(f"   Red channel std: {r_std:.1f}")
            print(f"   Green channel std: {g_std:.1f}")
            print(f"   Blue channel std: {b_std:.1f}")
            print(f"   Has variation: {has_variation}")
            print(f"   Is colorful: {is_colorful}")
            
            if has_variation and is_colorful:
                print(f"✅ Normal map looks correct - colorful with normal data")
            else:
                print(f"⚠️  Normal map might be diffuse render - not colorful enough")
                print(f"   Expected: Colorful gradients showing surface normals")
                print(f"   Got: Possibly grayscale or flat diffuse render")
        else:
            print(f"⚠️  Normal map is not RGB - might be grayscale")
            
    except Exception as e:
        print(f"❌ Error checking normal map: {e}")

def check_albedo_map_quality(output_dir, envmap):
    """Check if albedo maps are properly rendered"""
    import imageio
    import numpy as np
    
    # Find the albedo map subdirectory
    albedo_subdir = os.path.join(output_dir, f"albedo_{envmap}_")
    
    if not os.path.exists(albedo_subdir):
        print(f"⚠️  No albedo map directory found: {albedo_subdir}")
        return
    
    # Find the first albedo map file
    albedo_files = [f for f in os.listdir(albedo_subdir) if f.startswith("Image") and f.endswith('.png')]
    
    if not albedo_files:
        print(f"⚠️  No albedo map files found in {albedo_subdir}")
        return
    
    # Check the first albedo map file
    albedo_file = os.path.join(albedo_subdir, albedo_files[0])
    print(f"🔍 Checking albedo map quality: {albedo_file}")
    
    try:
        # Load the image
        img = imageio.imread(albedo_file)
        
        if len(img.shape) == 3 and img.shape[2] >= 3:  # RGB or RGBA
            # Check if it has color variation (indicating material differences)
            r_channel = img[:, :, 0]
            g_channel = img[:, :, 1] 
            b_channel = img[:, :, 2]
            
            # Calculate statistics
            r_std = np.std(r_channel)
            g_std = np.std(g_channel)
            b_std = np.std(b_channel)
            r_mean = np.mean(r_channel)
            g_mean = np.mean(g_channel)
            b_mean = np.mean(b_channel)
            
            # Check if it has reasonable color variation
            has_variation = r_std > 5 or g_std > 5 or b_std > 5
            
            # Check if it's not overly bright (should be realistic albedo values)
            is_realistic_brightness = r_mean < 200 and g_mean < 200 and b_mean < 200
            
            print(f"📊 Albedo map analysis:")
            print(f"   Red channel: mean={r_mean:.1f}, std={r_std:.1f}")
            print(f"   Green channel: mean={g_mean:.1f}, std={g_std:.1f}")
            print(f"   Blue channel: mean={b_mean:.1f}, std={b_std:.1f}")
            print(f"   Has variation: {has_variation}")
            print(f"   Realistic brightness: {is_realistic_brightness}")
            
            if has_variation and is_realistic_brightness:
                print(f"✅ Albedo map looks correct - shows material colors with realistic values")
            elif not has_variation:
                print(f"⚠️  Albedo map might be too uniform")
                print(f"   Expected: Color variation showing different materials")
                print(f"   Got: Possibly uniform or empty render")
            elif not is_realistic_brightness:
                print(f"⚠️  Albedo map might be too bright")
                print(f"   Expected: Realistic material colors (not emission-like)")
                print(f"   Got: Possibly overly bright or emission-like values")
        else:
            print(f"⚠️  Albedo map is not RGB - might be grayscale")
            
    except Exception as e:
        print(f"❌ Error checking albedo map: {e}")

def process_single_model(model_path, output_root, envmaps):
    """Process a single model: check materials, render, and create MP4"""
    file_name = os.path.splitext(os.path.basename(model_path))[0]
    
    print(f"\n🎬 Processing: {file_name}")
    print("=" * 50)
    
    # Step 1: Clear scene and import model
    clear_scene()
    print(f"📁 Importing: {model_path}")
    
    if not import_model(model_path):
        print(f"❌ Failed to import: {model_path}")
        return {"error": "Import failed"}
    
    # Step 2: Check materials
    print("🔍 Checking materials...")
    material_info = check_materials()
    print(f"📊 Found {material_info['num_materials']} materials")
    print(f"📋 Materials: {material_info['materials']}")
    
    # Step 3: Check if model has materials
    if material_info['num_materials'] < MATERIAL_CHECK_SETTINGS['min_materials']:
        print(f"⏭️  Model has insufficient materials - skipping render")
        return {"material_info": material_info, "rendered": False}
    
    # Step 4: Get mesh objects and setup
    mesh_objects = [obj for obj in bpy.data.objects if obj.type == 'MESH']
    if not mesh_objects:
        print("❌ No mesh objects found")
        return {"error": "No mesh objects", "material_info": material_info}
    
    target_object = mesh_objects[0]
    print(f"🎯 Target object: {target_object.name}")
    
    # Step 5: Center and scale objects
    print("🔄 Centering and scaling objects...")
    center_and_scale_objects(mesh_objects, target_size=CAMERA_SETTINGS['target_object_size'])
    
    # Step 6: Render for each environment
    rendered_envmaps = []
    
    for envmap in envmaps:
        print(f"\n🎬 Rendering with {envmap} environment...")
        
        # Setup output directory
        output_dir = os.path.join(output_root, file_name, envmap)
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup camera and lighting
        hdr_path = os.path.join(BLENDER_ROOT, "3.6", "datafiles", "studiolights", "world", f"{envmap}.exr")
        if not os.path.exists(hdr_path):
            hdr_path = None  # Use simple lighting if HDR not found
        
        setup_camera_and_lighting(target_object, hdr_path)
        
        # # Setup render settings
        # setup_render_settings(f"{output_dir}/frame_", RENDER_SETTINGS['resolution_x'], RENDER_SETTINGS['resolution_y'])

        # Setup render settings for regular rendering
        setup_render_settings(f"{output_dir}/frame_", RENDER_SETTINGS['resolution_x'], RENDER_SETTINGS['resolution_y'], render_normal_maps=False)
        
        # Set animation range
        scene = bpy.context.scene
        scene.frame_start = 1
        scene.frame_end = RENDER_SETTINGS['frames']
        
        # Render regular animation FIRST (with proper materials)
        print(f"🎬 Rendering {RENDER_SETTINGS['frames']} frames...")
        bpy.ops.render.render(animation=True)
        
        # Store a copy of the regular frames before any material changes
        regular_frames_dir = os.path.join(output_dir, "regular_frames")
        os.makedirs(regular_frames_dir, exist_ok=True)
        
        # Copy regular frames to safe location
        for i in range(1, RENDER_SETTINGS['frames'] + 1):
            src_frame = os.path.join(output_dir, f"frame_{i:04d}.png")
            dst_frame = os.path.join(regular_frames_dir, f"frame_{i:04d}.png")
            if os.path.exists(src_frame):
                import shutil
                shutil.copy2(src_frame, dst_frame)
        
        # Render normal maps with Eevee (if enabled)
        if OUTPUT_SETTINGS['render_normal_maps']:
            print(f"🎨 Rendering normal maps for {envmap}...")
            render_normal_maps_with_eevee(mesh_objects, output_dir, envmap)
        else:
            print(f"⏭️  Normal map rendering disabled for {envmap}")
        
        # Render albedo maps with Eevee (if enabled)
        if OUTPUT_SETTINGS['render_albedo_maps']:
            print(f"🎨 Rendering albedo maps for {envmap}...")
            render_albedo_maps_with_eevee(mesh_objects, output_dir, envmap)
        else:
            print(f"⏭️  Albedo map rendering disabled for {envmap}")
        
        # Restore regular frames from safe location
        for i in range(1, RENDER_SETTINGS['frames'] + 1):
            src_frame = os.path.join(regular_frames_dir, f"frame_{i:04d}.png")
            dst_frame = os.path.join(output_dir, f"frame_{i:04d}.png")
            if os.path.exists(src_frame):
                import shutil
                shutil.copy2(src_frame, dst_frame)
        
        # Clean up temporary directory
        import shutil
        shutil.rmtree(regular_frames_dir, ignore_errors=True)
        
        # Create MP4s for different map types
        if OUTPUT_SETTINGS['create_mp4']:
            # Create organized video directories
            model_videos_dir = os.path.join(output_root, "model_videos")
            normal_videos_dir = os.path.join(output_root, "normal_map_videos")
            albedo_videos_dir = os.path.join(output_root, "albedo_videos")
            
            os.makedirs(model_videos_dir, exist_ok=True)
            os.makedirs(normal_videos_dir, exist_ok=True)
            os.makedirs(albedo_videos_dir, exist_ok=True)
            
            # 1. Create regular model MP4
            model_mp4_path = os.path.join(model_videos_dir, f"{file_name}_{envmap}.mp4")
            
            # Check if frames exist
            frame_exists = any(os.path.exists(os.path.join(output_dir, f"frame_{i:04d}.png")) 
                             for i in range(1, RENDER_SETTINGS['frames'] + 1))
            
            if frame_exists:
                try:
                    # Use FFMPEG for better MP4 creation
                    cmd = [
                        'ffmpeg',
                        '-y',  # Overwrite output file
                        '-framerate', '30',
                        '-i', os.path.join(output_dir, 'frame_%04d.png'),
                        '-c:v', 'libx264',
                        '-pix_fmt', 'yuv420p',
                        '-crf', '23',
                        model_mp4_path
                    ]
                    
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        print(f"✅ Created model MP4: {model_mp4_path}")
                        rendered_envmaps.append(envmap)
                    else:
                        print(f"⚠️  FFMPEG failed, trying imageio fallback...")
                        # Fallback to imageio
                        imgs = []
                        for i in range(1, RENDER_SETTINGS['frames'] + 1):
                            frame_path = os.path.join(output_dir, f"frame_{i:04d}.png")
                            if os.path.exists(frame_path):
                                img = imageio.imread(frame_path)
                                if img.shape[2] == 4:  # RGBA
                                    rgb, mask = img[:, :, :3], img[:, :, 3:4]
                                    img = rgb * mask + (1 - mask) * 255
                                imgs.append(img)
                        
                        if imgs:
                            try:
                                imageio.mimsave(model_mp4_path, imgs)
                                rendered_envmaps.append(envmap)
                                print(f"✅ Created model MP4 with imageio: {model_mp4_path}")
                            except Exception as e:
                                print(f"Warning: Could not create model MP4: {e}")
                                # Still count as rendered since frames exist
                                rendered_envmaps.append(envmap)
                except Exception as e:
                    print(f"Warning: Could not create model MP4: {e}")
                    # Still count as rendered since frames exist
                    rendered_envmaps.append(envmap)
            else:
                print(f"❌ No frames found in {output_dir}")
            
            # 2. Create normal map MP4 (if normal maps were rendered)
            if OUTPUT_SETTINGS['render_normal_maps']:
                normal_mp4_path = os.path.join(normal_videos_dir, f"{file_name}_{envmap}_normal.mp4")
                
                # Check if normal map frames exist in the subdirectory
                normal_subdir = os.path.join(output_dir, f"normal_{envmap}_")
                normal_frame_exists = False
                
                if os.path.exists(normal_subdir):
                    # Check if Image files exist in the subdirectory
                    normal_frame_exists = any(os.path.exists(os.path.join(normal_subdir, f"Image{i:04d}.png")) 
                                            for i in range(1, RENDER_SETTINGS['frames'] + 1))
                
                if normal_frame_exists:
                    try:
                        cmd = [
                            'ffmpeg',
                            '-y',
                            '-framerate', '30',
                            '-i', os.path.join(normal_subdir, 'Image%04d.png'),
                            '-c:v', 'libx264',
                            '-pix_fmt', 'yuv420p',
                            '-crf', '23',
                            normal_mp4_path
                        ]
                        
                        result = subprocess.run(cmd, capture_output=True, text=True)
                        
                        if result.returncode == 0:
                            print(f"✅ Created normal map MP4: {normal_mp4_path}")
                        else:
                            print(f"⚠️  Failed to create normal map MP4: {result.stderr}")
                    except Exception as e:
                        print(f"Warning: Could not create normal map MP4: {e}")
                else:
                    print(f"❌ No normal map frames found in {normal_subdir}")
            
            # 3. Create albedo map MP4 (if albedo maps were rendered)
            if OUTPUT_SETTINGS['render_albedo_maps']:
                albedo_mp4_path = os.path.join(albedo_videos_dir, f"{file_name}_{envmap}_albedo.mp4")
                
                # Check if albedo map frames exist in the subdirectory
                albedo_subdir = os.path.join(output_dir, f"albedo_{envmap}_")
                albedo_frame_exists = False
                
                if os.path.exists(albedo_subdir):
                    # Check if Image files exist in the subdirectory
                    albedo_frame_exists = any(os.path.exists(os.path.join(albedo_subdir, f"Image{i:04d}.png")) 
                                            for i in range(1, RENDER_SETTINGS['frames'] + 1))
                
                if albedo_frame_exists:
                    try:
                        cmd = [
                            'ffmpeg',
                            '-y',
                            '-framerate', '30',
                            '-i', os.path.join(albedo_subdir, 'Image%04d.png'),
                            '-c:v', 'libx264',
                            '-pix_fmt', 'yuv420p',
                            '-crf', '23',
                            albedo_mp4_path
                        ]
                        
                        result = subprocess.run(cmd, capture_output=True, text=True)
                        
                        if result.returncode == 0:
                            print(f"✅ Created albedo map MP4: {albedo_mp4_path}")
                        else:
                            print(f"⚠️  Failed to create albedo map MP4: {result.stderr}")
                    except Exception as e:
                        print(f"Warning: Could not create albedo map MP4: {e}")
                else:
                    print(f"❌ No albedo map frames found in {albedo_subdir}")
        else:
            # Just check if frames were created
            frame_exists = any(os.path.exists(os.path.join(output_dir, f"frame_{i:04d}.png")) 
                             for i in range(1, RENDER_SETTINGS['frames'] + 1))
            if frame_exists:
                rendered_envmaps.append(envmap)
    
    return {
        "material_info": material_info,
        "rendered": True,
        "rendered_envmaps": rendered_envmaps,
        "file_name": file_name
    }

def process_directory(input_dir, output_root, envmaps=None):
    """Process all 3D models in a directory"""
    if envmaps is None:
        envmaps = ENVIRONMENT_MAPS
    
    # Find all 3D model files
    model_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in FILE_EXTENSIONS):
                model_files.append(os.path.join(root, file))
    
    print(f"📁 Found {len(model_files)} 3D model files to process")
    
    
    # Results tracking
    results = {
        'total_models': len(model_files),
        'models_with_materials': [],
        'models_without_materials': [],
        'rendered_models': [],
        'failed_renders': [],
        'errors': []
    }
    
    # Process each model
    for model_path in tqdm(model_files, desc="Processing models"):
        try:
            result = process_single_model(model_path, output_root, envmaps)
            
            if "error" in result:
                print(f"❌ Error processing {os.path.basename(model_path)}: {result['error']}")
                results['errors'].append({
                    'file': model_path,
                    'error': result['error']
                })
                continue
            
            if result['rendered']:
                print(f"✅ Successfully processed: {result['file_name']}")
                results['models_with_materials'].append({
                    'file': model_path,
                    'material_info': result['material_info']
                })
                results['rendered_models'].append({
                    'file': model_path,
                    'rendered_envmaps': result['rendered_envmaps']
                })
            else:
                print(f"⏭️  Skipped rendering: {os.path.basename(model_path)} (no materials)")
                results['models_without_materials'].append(model_path)
                
        except Exception as e:
            print(f"❌ Exception processing {os.path.basename(model_path)}: {str(e)}")
            results['errors'].append({
                'file': model_path,
                'error': str(e)
            })
    
    return results

def save_results(results, output_dir):
    """Save processing results to files"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_models': results['total_models'],
        'models_with_materials': len(results['models_with_materials']),
        'models_without_materials': len(results['models_without_materials']),
        'rendered_models': len(results['rendered_models']),
        'failed_renders': len(results['failed_renders']),
        'errors': len(results['errors'])
    }
    
    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save detailed results
    with open(os.path.join(output_dir, 'detailed_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save CSV report
    csv_path = os.path.join(output_dir, 'processing_report.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['File', 'Has Materials', 'Rendered', 'Rendered Environments', 'Error'])
        
        for model in results['models_with_materials']:
            rendered_info = next((r for r in results['rendered_models'] if r['file'] == model['file']), None)
            rendered = 'Yes' if rendered_info else 'No'
            envmaps = ', '.join(rendered_info['rendered_envmaps']) if rendered_info else ''
            writer.writerow([model['file'], 'Yes', rendered, envmaps, ''])
        
        for model in results['models_without_materials']:
            writer.writerow([model, 'No', 'No', '', ''])
        
        for error in results['errors']:
            writer.writerow([error['file'], 'Unknown', 'No', '', error['error']])
    
    print(f"\n📊 Results saved to {output_dir}")
    print(f"📋 Summary: {summary}")

def main():
    """Main function to run the complete standalone pipeline"""
    print("🎬 Complete Standalone Pipeline")
    print("=" * 60)
    print(f"📁 Input directory: {INPUT_DIR}")
    print(f"📁 Output directory: {OUTPUT_ROOT}")
    print(f"📁 Results directory: {RESULTS_DIR}")
    print(f"🎨 Environment maps: {ENVIRONMENT_MAPS}")
    print(f"🎬 Render settings: {RENDER_SETTINGS['frames']} frames, {RENDER_SETTINGS['resolution_x']}x{RENDER_SETTINGS['resolution_y']}")
    print("=" * 60)
    
    # Check if input directory exists
    if not os.path.exists(INPUT_DIR):
        print(f"❌ Input directory does not exist: {INPUT_DIR}")
        print("💡 Please modify the INPUT_DIR variable at the top of the script")
        return
    
    # Create output directories
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Process all models
    results = process_directory(INPUT_DIR, OUTPUT_ROOT, ENVIRONMENT_MAPS)
    
    # Save results
    save_results(results, RESULTS_DIR)

    # Create video compositions
    create_video_compositions(results['rendered_models'], OUTPUT_ROOT)
    
    # Print final summary
    print("\n" + "="*60)
    print("🎬 PIPELINE COMPLETED")
    print("="*60)
    print(f"📊 Total models processed: {results['total_models']}")
    print(f"✅ Models with materials: {len(results['models_with_materials'])}")
    print(f"⏭️  Models without materials: {len(results['models_without_materials'])}")
    print(f"🎬 Successfully rendered: {len(results['rendered_models'])}")
    print(f"❌ Failed renders: {len(results['failed_renders'])}")
    print(f"⚠️  Errors: {len(results['errors'])}")
    print("="*60)

if __name__ == "__main__":
    main() 
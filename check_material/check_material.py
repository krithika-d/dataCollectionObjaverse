import bpy
import sys
import json
import os
from pathlib import Path

def load_mesh(filepath):
    ext = filepath.split('.')[-1].lower()
    if ext == 'obj':
        bpy.ops.import_scene.obj(filepath=filepath)
    elif ext == 'fbx':
        bpy.ops.import_scene.fbx(filepath=filepath)
    elif ext in ['glb', 'gltf']:
        bpy.ops.import_scene.gltf(filepath=filepath)
    else:
        raise ValueError(f"Unsupported format: {ext}")

def get_hierarchy_path(obj):
    path = []
    current = obj
    while current:
        path.insert(0, current.name)
        current = current.parent
    return "/".join(path)

def trace_destination_sockets(socket, visited=None):
    if visited is None:
        visited = set()
    connected_names = []

    for link in socket.links:
        target_node = link.to_node
        target_socket = link.to_socket

        # Prevent cycles
        if (target_node, target_socket) in visited:
            continue
        visited.add((target_node, target_socket))

        if target_node.type == 'BSDF_PRINCIPLED':
            connected_names.append(target_socket.name)
        else:
            for output in target_node.outputs:
                connected_names.extend(trace_destination_sockets(output, visited))

    return connected_names

def analyze_scene(filepath):
    mesh_objects = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']
    all_materials = set()
    object_summaries = []

    for obj in mesh_objects:
        obj_data = {
            "name": obj.name,
            "hierarchy_path": get_hierarchy_path(obj),
            "materials": []
        }

        for slot in obj.material_slots:
            mat = slot.material
            if not mat:
                continue
            all_materials.add(mat.name)

            mat_info = {
                "name": mat.name,
                "has_nodes": mat.use_nodes,
                "type": None,
                "textures": []  # now holds multiple mappings
            }

            if mat.use_nodes:
                for node in mat.node_tree.nodes:
                    if node.type == 'BSDF_PRINCIPLED':
                        mat_info["type"] = "Principled BSDF"
                        break  # optional: stop at first found BSDF

                for node in mat.node_tree.nodes:
                    if node.type == 'TEX_IMAGE':
                        image = node.image
                        image_name = image.name if image else "None"
                        connected_to = []
                        texture_size = None
                        image_filepath = None

                        # Trace final connections
                        for output in node.outputs:
                            connected_to.extend(trace_destination_sockets(output))

                        if image:
                            try:
                                width, height = image.size
                                if width > 0 and height > 0:
                                    texture_size = f"{width}x{height}"
                                else:
                                    # Try resolving path relative to model location if size is zero
                                    if image.filepath and not image.packed_file:
                                        mesh_dir = os.path.dirname(filepath)
                                        candidate_path = os.path.join(mesh_dir, image.filepath)

                                        if os.path.exists(candidate_path):
                                            try:
                                                new_image = bpy.data.images.load(candidate_path, check_existing=True)
                                                image = new_image
                                                width, height = image.size
                                                if width > 0 and height > 0:
                                                    texture_size = f"{width}x{height}"
                                            except:
                                                pass

                                # Convert image path to absolute and then make it relative to the model file
                                abs_texture_path = bpy.path.abspath(image.filepath)
                                image_filepath = os.path.relpath(abs_texture_path, filepath)
                            except:
                                pass

                        # TODO: also need to locate when the material texture is available but not successfully loaded, use "?" maybe

                        mat_info["textures"].append({
                            "texture_name": image_name,
                            "connected_to": list(set(connected_to)),
                            "texture_size": texture_size,
                            "image_filepath": image_filepath
                        })

            obj_data["materials"].append(mat_info)
        object_summaries.append(obj_data)

    result = {
        "num_parts": len(mesh_objects),
        "num_materials": len(all_materials),
        "objects": object_summaries
    }
    return result

def main():
    filepath = sys.argv[-1]
    load_mesh(filepath)
    result = analyze_scene(filepath)
    print(json.dumps(result))

if __name__ == "__main__":
    main()

import bpy
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'blenderkit')))
# ─── Ensure BlenderKit addon modules are importable ───────────────────────────
# for script_path in bpy.utils.script_paths():
#     addon_path = os.path.join(script_path, "addons", "blenderkit")
#     if os.path.isdir(addon_path):
#         sys.path.append(addon_path)
#         break
from blenderkit import search, download
bpy.ops.preferences.addon_enable(module="blenderkit")
# ─── Configuration ───────────────────────────────────────────────────────────
MAX_MODELS = 1
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, 'dataset')
os.makedirs(DATASET_DIR, exist_ok=True)
# ─── Helper: completely wipe all objects and orphan data ──────────────────────
def clear_scene():
    # Remove all objects
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    # Remove orphan meshes
    for mesh in list(bpy.data.meshes):
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)
# ─── Build search parameters ──────────────────────────────────────────────────
params = {
    'asset_type': 'model',
    'is_free': 'true',
}
# ─── Query BlenderKit for free models ─────────────────────────────────────────
print("Querying free models…")
results = search.get_search_simple(params, page_size=100, max_results=MAX_MODELS)
print(f"Found {len(results)} free models (limiting to {MAX_MODELS}).")
# ─── Loop: download, convert instances, export FBX ────────────────────────────
for idx, asset_data in enumerate(results[:MAX_MODELS], start=1):
    aid  = asset_data.get('id', f'asset_{idx}')
    name = asset_data.get('name', asset_data.get('displayName', aid))
    print(f"[{idx}/{MAX_MODELS}] Processing: {name}")
    # 1) clear previous scene
    clear_scene()
    # 2) record pre-download objects
    before = set(bpy.data.objects)
    # 3) download & append into scene
    try:
        download.start_download(
            asset_data,
            model_location=(0, 0, 0),
            model_rotation=(0, 0, 0),
            resolution='ORIGINAL'
        )
    except Exception as e:
        print(f"  :warning: Download failed for {name}: {e}")
        continue
    # 3.5) Wait for new objects to appear (download is async)
    import time
    timeout = 30  # seconds
    poll_interval = 1
    start_time = time.time()
    while True:
        after = set(bpy.data.objects)
        new_objects = after - before
        if new_objects or (time.time() - start_time) > timeout:
            break
        time.sleep(poll_interval)
    # Deselect everything
    bpy.ops.object.select_all(action='DESELECT')
    # Select all objects in all collections in the scene (including nested)
    def select_all_objects_in_scene_collections():
        for collection in bpy.context.scene.collection.children_recursive:
            for obj in collection.objects:
                obj.select_set(True)
        # Also select objects directly in the master scene collection
        for obj in bpy.context.scene.collection.objects:
            obj.select_set(True)
    select_all_objects_in_scene_collections()
    # Export the entire scene as FBX (no selection needed)
    fbx_path = os.path.join(DATASET_DIR, f"{aid}.fbx")
    bpy.ops.export_scene.fbx(
        filepath=fbx_path,
        use_selection=False,
        embed_textures=True,
        path_mode='COPY'
    )
    print(f"  :white_check_mark: Exported FBX to {fbx_path}")
print("All done.")
# run.py
import os
import sys
import json
import zipfile
import shutil
import tempfile
import argparse
import subprocess
from multiprocessing import Pool

BLENDER_PATH = "/home/jyang/projects/ObjectReal/external/objaverse-xl/scripts/rendering/blender-3.2.2-linux-x64/blender"  # e.g., "/usr/bin/blender"
CHECK_SCRIPT = os.path.abspath("/home/jyang/projects/ObjectReal/texnet/objverse/check_material.py")
MESH_DIR = "/path/to/meshes"       # Directory with mesh files or .zip files
MESH_EXTS = ['.obj', '.fbx', '.glb']

def extract_meshes_from_zip(zip_path, extract_dir):
    mesh_files = []
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
        for root, _, files in os.walk(extract_dir):
            for file in files:
                if os.path.splitext(file)[1].lower() in MESH_EXTS:
                    mesh_files.append(os.path.join(root, file))
    return mesh_files

def get_all_mesh_files(root_dir):
    all_meshes = []
    for f in os.listdir(root_dir):
        full_path = os.path.join(root_dir, f)
        ext = os.path.splitext(f)[1].lower()
        if ext in MESH_EXTS:
            all_meshes.append(full_path)
        elif ext == ".zip":
            extract_dir = tempfile.mkdtemp(prefix="mesh_zip_")
            meshes_in_zip = extract_meshes_from_zip(full_path, extract_dir)
            for mesh in meshes_in_zip:
                all_meshes.append((mesh, extract_dir))  # track temp folder for cleanup
        else:
            continue
    return all_meshes

def run_blender_check(entry):
    if isinstance(entry, tuple):
        mesh_file, tmp_dir = entry
    else:
        mesh_file = entry
        tmp_dir = None

    try:
        result = subprocess.run([
            BLENDER_PATH, "-b", "--python", CHECK_SCRIPT, "--", mesh_file
        ], capture_output=True, text=True, timeout=120)

        if result.returncode != 0:
            return {"file": mesh_file, "error": result.stderr.strip()}
        
        if 'error' in result.stdout.lower():
            return {"file": mesh_file, "error": result.stdout.strip()}

        output = result.stdout.strip().splitlines()
        try:
            # find the line that contains the JSON output
            output = [line for line in output if line.startswith("{") or line.startswith("[")]
            data = json.loads(output[0])
            return {"file": mesh_file, "materials": data}
        except json.JSONDecodeError:
            return {"file": mesh_file, "error": f"Failed to parse JSON: {result.stdout.strip()}"}
        except IndexError:
            return {"file": mesh_file, "error": "No JSON output found in Blender script output."}
    except subprocess.TimeoutExpired:
        return {"file": mesh_file, "error": "Timeout"}
    finally:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)

def main(use_multiprocessing=True):
    # all_entries = get_all_mesh_files(MESH_DIR)
    root = '/labworking/Users/jyang/data/objaverse/github/repos/'
    all_entries = [
        # '/home/jyang/projects/ObjectReal/data/check_material/autumn_house.glb'
        # root + 'MasterPuffin/HSFL_Projekt/main-build/Phase Alpha/Assets/3D Assets/Stones/Stone3/stone3-L.fbx'
        root + 'MasterPuffin/HSFL_Projekt/main-build/Phase Alpha/Assets/3D Assets/spaceship broken/spaceshipmodel4old.fbx'
        # root + 'MasterPuffin/HSFL_Projekt/main-build/Phase Alpha/Assets/3D Assets/Alien Technology/cogwheel_large.fbx'
    ]
    if use_multiprocessing:
        with Pool(processes=os.cpu_count()) as pool:
            results = pool.map(run_blender_check, all_entries)
    else:
        results = [run_blender_check(e) for e in all_entries]

    with open("analysis_output/material_debug_report.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Material check finished. Results saved to material_debug_report.json.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Blender mesh material checker (supports .zip).")
    parser.add_argument("--no-mp", action="store_true", help="Disable multiprocessing")
    args = parser.parse_args()
    args.no_mp = True
    main(use_multiprocessing=not args.no_mp)

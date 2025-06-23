import os
import zipfile
import csv
import json
import shutil
import subprocess
import tempfile
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict
from multiprocessing import Pool
import time

FILE_EXTENSIONS = [
    ".obj", ".glb", ".gltf", ".usdz", ".usd", ".fbx",
    ".stl", ".usda", ".dae", ".ply", ".abc", ".blend"
]

MATERIAL_EXTENSIONS = [
    ".mtl", ".jpeg", ".jpg", ".png", ".bmp", ".tga", ".gif", '.hdr', ".exr", ".tif", ".tiff", ".webp", ".svg", ".psd",
]

INVALID_EXTENSIONS = [
    ".exe", ".dll", ".bat", ".sh", ".py", ".pyc", ".pdb", ".log", ".tmp", ".ini"
]

BLENDER_PATH = "/home/jyang/projects/ObjectReal/external/objaverse-xl/scripts/rendering/blender-3.2.2-linux-x64/blender"  # Update as needed
CHECK_SCRIPT = "/home/jyang/projects/ObjectReal/texnet/objverse/check_material.py"  # Update as needed

def run_blender_check(mesh_path):
    # use run_check_material_debug to check the detailed output of the CHECK_SCRIPT, which should be a json format
    try:
        result = subprocess.run(
            [BLENDER_PATH, "-b", "--python", CHECK_SCRIPT, "--", mesh_path],
            capture_output=True, text=True, timeout=60
        )
        output = result.stdout.strip().splitlines()
        output = [line for line in output if line.startswith("{") or line.startswith("[")]
        material_info = json.loads(output[0])
        return {"filename": os.path.basename(mesh_path), 'fullpath': mesh_path, "material_info": material_info}
    except Exception as e:
        return {"filename": os.path.basename(mesh_path), 'fullpath': mesh_path, "error": str(e)}

def analyze_zip(zip_path, use_multiprocess=True):
    type_counts = defaultdict(int)
    material_type_counts = defaultdict(int)
    invalid_type_counts = defaultdict(int)

    invalid_files = []
    models_with_materials = []
    models_without_materials = []
    model_meta = []

    total_models = 0
    has_material = False

    texture_usage_counts = defaultdict(int)
    model_texture_table = []

    extract_dir = zip_path.replace('.zip', '') # don't use prefix to shorten the path length

    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:

            # get all valid files in the zip
            members = zf.namelist()
            valid_files = [
                f for f in members
                if os.path.splitext(f)[1].lower() in FILE_EXTENSIONS + MATERIAL_EXTENSIONS
            ]

            # Check if all valid files already exist
            already_exist = all(
                os.path.exists(os.path.join(extract_dir, *f.split('/')))
                for f in valid_files
            )

            if not already_exist:
                return None
                os.makedirs(extract_dir, exist_ok=True)
                print(f"  - Extracting {len(valid_files)}/{len(members)} files from {zip_path}")
                try:
                    zf.extractall(path=extract_dir, members=valid_files) # extractall is vectorized
                    print(f"  - Files extraction to {extract_dir} completed: {len(valid_files)} files extracted")
                except Exception as e:
                    print(f"Error during batch extraction from {zip_path}: {e}")
            else:
                print(f"  - All files already exist in {extract_dir}. Skipping extraction.")

            model_paths = []
            model_names = []
            for root, _, files in os.walk(extract_dir):
                for file in files:
                    ext = os.path.splitext(file)[1].lower()
                    full_path = os.path.join(root, file)
                    if ext in FILE_EXTENSIONS:
                        model_paths.append(full_path)
                        model_names.append(file)
                        total_models += 1
                        type_counts[ext] += 1
                    elif ext in MATERIAL_EXTENSIONS:
                        material_type_counts[ext] += 1
                    elif ext in INVALID_EXTENSIONS:
                        invalid_files.append(file)
                        invalid_type_counts[ext] += 1

            print(f'  - Processing {len(model_paths)} 3D model files from {extract_dir}. Multiprocessing enabled: {use_multiprocess}')
            if use_multiprocess:
                # with Pool(processes=2) as pool: # 0.81s/model (238 models)
                # with Pool(processes=4) as pool: # 0.70s/model (238 models)
                # with Pool(processes=8) as pool: # 0.74s/model (238 models)
                # with Pool(processes=16) as pool: # 0.63s/model (238 models)
                # with Pool(processes=32) as pool: # 0.64s/model (238 models)
                # with Pool(processes=64) as pool: # 0.60s/model (238 models)
                with Pool(processes=128) as pool: # 0.64s/model (238 models)
                # with Pool(processes=256) as pool: # 0.61s/model (238 models)
                    results = list(tqdm(pool.imap_unordered(run_blender_check, model_paths), total=len(model_paths), desc="  - Checking material/texture via Blender"))
            else:
                results = [run_blender_check(path) for path in tqdm(model_paths, desc="  - Checking material/texture via Blender")] # 1.12s/model (238 models)

            for result in results:
                fname = result["filename"]
                fpath_relative_to_repo = os.path.relpath(
                    result["fullpath"], start=os.path.dirname(extract_dir)
                )
                if "error" in result:
                    models_without_materials.append(fname)
                    model_meta.append({"filename": fname, "error": result["error"]})
                    continue

                material_info = result["material_info"]
                has_mat = material_info.get("num_materials", 0) > 0
                if has_mat:
                    models_with_materials.append(fname)
                    has_material = True
                else:
                    models_without_materials.append(fname)

                model_meta.append({
                    "filename": fname,
                    'filepath_rel': fpath_relative_to_repo,
                    "num_parts": material_info.get("num_parts", 0),
                    "num_materials": material_info.get("num_materials", 0),
                    "objects": material_info.get("objects", [])
                })

                texture_flags = defaultdict(bool)
                for obj in material_info.get("objects", []):
                    for mat in obj.get("materials", []):
                        for tex in mat.get("textures", []):
                            for socket in tex.get("connected_to", []):
                                texture_flags[socket] = True
                                if tex.get("texture_size"):
                                    # use the texture size if available
                                    texture_flags[socket + '_size'] = tex["texture_size"]
                                elif tex.get("image_filepath").replace('.', '').replace('/', ''):
                                    # use 0x0 indiates the texture is available but not found
                                    texture_flags[socket + '_size'] = f'?x?'
                                else:
                                    # if no texture, set the size to empty
                                    texture_flags[socket + '_size'] = ''
                                texture_usage_counts[socket] += 1

                model_texture_table.append({
                    "filename": fname,
                    'filepath_rel': fpath_relative_to_repo,
                    'num_parts': material_info.get("num_parts", 0),
                    'num_materials': material_info.get("num_materials", 0),
                    "texture_flags": dict(texture_flags)
                })

    except Exception as e:
        print(f"Error reading {zip_path}: {e}")
        return None

    return {
        'total_models': total_models,
        'type_counts': dict(type_counts),
        'has_material': has_material,
        'material_type_counts': dict(material_type_counts),
        'invalid_files': invalid_files,
        'invalid_type_counts': dict(invalid_type_counts),
        'models_with_materials': models_with_materials,
        'models_without_materials': models_without_materials,
        'model_meta': model_meta,
        'texture_usage_counts': dict(texture_usage_counts),
        'model_texture_table': model_texture_table,
    }

def save_asset_texture_table(src, stats, out_dir="analysis_output"):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{src}_asset_texture_summary.csv")

    texture_keys = set()
    for row in stats['asset_texture_rows']:
        keys2skip = ["repo", "filename", "filepath_rel", "num_parts", "num_materials"]
        texture_keys.update(k for k in row.keys() if k not in keys2skip)
    texture_keys = sorted(texture_keys)
    texture_channels = sorted(set(k for k in texture_keys if not k.endswith("_size")))
    header = ["Filepath", "#Parts", "#Materials"] + texture_channels

    # sort the stats['asset_texture_rows'] by repo and filepath_rel
    stats['asset_texture_rows'].sort(key=lambda x: (x['repo'], x['filepath_rel']))

    with open(path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in stats['asset_texture_rows']:
            writer.writerow([
                row["repo"] + "/" + row["filepath_rel"],
                row['num_parts'],
                row['num_materials'],
                # *[int(row.get(k, False)) for k in texture_channels] # 1 or 0
                *[str(row.get(k+'_size', '')) for k in texture_channels]
            ])

def extract_zip_if_needed(zip_info):
    zip_path, extract_dir = zip_info

    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            members = zf.namelist()
            valid_files = [
                f for f in members
                if os.path.splitext(f)[1].lower() in FILE_EXTENSIONS + MATERIAL_EXTENSIONS
            ]

            # Check if all valid files already exist
            already_exist = all(
                os.path.exists(os.path.join(extract_dir, *f.split('/')))
                for f in valid_files
            )

            if already_exist:
                return zip_path, False  # Skip extraction

            os.makedirs(extract_dir, exist_ok=True)
            zf.extractall(path=extract_dir, members=valid_files)
        return zip_path, True
    except Exception as e:
        print(f"  [Error] Failed to extract {zip_path}: {e}")
        return zip_path, False

def pre_extract_all_zips(base_path, max_repos=10, num_workers=32):
    zip_tasks = []

    with os.scandir(base_path) as entries:
        entries = [e for e in entries if e.is_dir()]
        print(f'[Preprocessing] Found {len(entries)} repositories in {os.path.basename(base_path)}')
        if max_repos > 0:
            entries = entries[:max_repos]
            print(f'[Preprocessing] Unzipping up to {max_repos} repositories')
        for entry in entries:
            for root, _, files in os.walk(entry.path):
                for file in files:
                    if file.lower().endswith(".zip"):
                        zip_path = os.path.join(root, file)
                        extract_dir = os.path.join(root, file[:-4])  # remove ".zip"
                        zip_tasks.append((zip_path, extract_dir))

    print(f"[Preprocessing] Found {len(zip_tasks)} ZIP files to extract")
    with Pool(processes=num_workers) as pool:
        list(tqdm(pool.imap_unordered(extract_zip_if_needed, zip_tasks),
                  total=len(zip_tasks), desc="  - Extracting ZIPs")) # tqdm update on each task completion

def analyze_all_zips(base_path, exclude_invalid_repos=None, use_multiprocess=True, pre_unzip=False):
    if exclude_invalid_repos is None:
        exclude_invalid_repos = set()

    max_repos = 7000
    if pre_unzip:
        # we can break the runtime anytime since the zips will be processed based on the completion of extraction
        pre_extract_all_zips(base_path, max_repos=max_repos, num_workers=128)

    stats = {
        'n_repos': 0,
        'n_zips': 0,
        'n_models': 0,
        'n_models_without_materials': 0,
        'repos_with_materials': 0,
        'model_type_counts': defaultdict(int),
        'material_type_counts': defaultdict(int),
        'invalid_file_count': 0,
        'repos_with_invalid_files': [],
        'repo_details': [],
        'texture_usage_counts': defaultdict(int),
        'asset_texture_rows': []
    }

    with os.scandir(base_path) as entries:
        # entries = [e for e in entries if e.is_dir()][:max_repos]
        entries = [e for e in entries if e.is_dir() and 'Canonelis' not in e.name][:30]
        tic = time.time()
        for i, entry in enumerate(tqdm(entries, desc=f'Analyzing ZIPs in {os.path.basename(base_path)}')):
            if entry.name in exclude_invalid_repos:
                continue

            repo_path = entry.path
            repo_models = 0
            repo_materials = 0
            has_material = False
            invalids_in_repo = []
            repo_type_counts = defaultdict(int)
            repo_material_type_counts = defaultdict(int)
            repo_invalid_type_counts = defaultdict(int)
            models_with_materials = []
            models_without_materials = []
            model_texture_table = []

            print(f" - repository {i}/{len(entries)}: {os.path.basename(entry.path)}")
            # repo_path = '/labworking/Users/jyang/data/objaverse/github/repos/zzxy0909'
            # repo_path = '/labworking/Users/jyang/data/objaverse/github/repos/NathanGrigne'
            unzipped_dir = []
            for root, dirs, files in os.walk(repo_path):

                # get all the unzipped directories
                for dir in dirs:
                    if dir + '.zip' in files:
                        unzipped_dir.append(os.path.join(root, dir))

                # loop through other files
                for file in files:

                    if any(root.startswith(unzip_dir) for unzip_dir in unzipped_dir):
                        # skip files in unzipped directories
                        continue
                    elif file.lower().endswith('.zip'):
                        # if 'HSFL_Projekt' in file: continue # debug
                        zip_path = os.path.join(root, file)
                        result = analyze_zip(zip_path, use_multiprocess=use_multiprocess)
                        if result:
                            stats['n_zips'] += 1
                            stats['n_models'] += result['total_models']
                            repo_models += result['total_models']
                            if result['has_material']:
                                has_material = True
                            repo_materials += sum(result['material_type_counts'].values())

                            for ext, count in result['type_counts'].items():
                                stats['model_type_counts'][ext] += count
                                repo_type_counts[ext] += count

                            for ext, count in result['material_type_counts'].items():
                                stats['material_type_counts'][ext] += count
                                repo_material_type_counts[ext] += count

                            if result['invalid_files']:
                                stats['invalid_file_count'] += len(result['invalid_files'])
                                invalids_in_repo.extend(result['invalid_files'])
                                for ext, count in result['invalid_type_counts'].items():
                                    repo_invalid_type_counts[ext] += count

                            models_with_materials.extend(result['models_with_materials'])
                            models_without_materials.extend(result['models_without_materials'])

                            for tex_type, count in result.get("texture_usage_counts", {}).items():
                                stats['texture_usage_counts'][tex_type] += count

                            for row in result.get("model_texture_table", []):
                                stats['asset_texture_rows'].append({
                                    "repo": entry.name,
                                    "filename": row["filename"],
                                    "filepath_rel": row["filepath_rel"],
                                    'num_parts': row.get("num_parts", 0),
                                    'num_materials': row.get("num_materials", 0),
                                    **row["texture_flags"]
                                })
                    elif file.lower().endswith(tuple(FILE_EXTENSIONS)):
                        # process files that not in zip nor unzipped
                        full_path = os.path.join(root, file)
                        ext = os.path.splitext(file)[1].lower()
                        stats['n_models'] += 1
                        repo_models += 1
                        stats['model_type_counts'][ext] += 1
                        repo_type_counts[ext] += 1

                        # Check for materials using Blender script
                        result = run_blender_check(full_path)
                        if "error" in result:
                            models_without_materials.append(file)
                            invalids_in_repo.append(file)
                            repo_invalid_type_counts["blender_error"] += 1
                        else:
                            models_with_materials.append(file)
                            has_material = True

            if repo_models > 0:
                stats['n_repos'] += 1
                if has_material:
                    stats['repos_with_materials'] += 1
                if repo_materials == 0:
                    stats['n_models_without_materials'] += repo_models
                stats['repo_details'].append({
                    'repo': entry.name,
                    'num_models': repo_models,
                    'model_types': dict(repo_type_counts),
                    'material_types': dict(repo_material_type_counts),
                    'has_material': has_material,
                    'models_with_materials': models_with_materials,
                    'models_without_materials': models_without_materials,
                    'invalid_files': invalids_in_repo,
                    'invalid_type_counts': dict(repo_invalid_type_counts)
                })
                if invalids_in_repo:
                    stats['repos_with_invalid_files'].append((entry.name, invalids_in_repo))

            toc = time.time()
            print(f"  - Analysis completed in {time.strftime('%H:%M:%S', time.gmtime(toc - tic))} for {stats['n_repos']} repos with {stats['n_models']} models.")

    return stats


def save_reports(src, stats, out_dir="analysis_output", write_model_paths=False):
    os.makedirs(out_dir, exist_ok=True)

    # sort the stats['repo_details'] by repo name
    stats['repo_details'].sort(key=lambda x: x['repo'])

    csv_path = os.path.join(out_dir, f"{src}_repo_analysis.csv")
    with open(csv_path, "w", newline='') as f:
        writer = csv.writer(f)
        header = [
            "Repo", "Num Models", "Model Types", "Material Types", "Has Material",
            "#Models With Materials", "#Models Without Materials",
            "Material Coverage (%)", "Invalid Count", "Invalid Types"
        ]
        if write_model_paths:
            header.insert(7, "Models With Materials")
            header.insert(8, "Models Without Materials")
        writer.writerow(header)

        for detail in stats['repo_details']:
            total = len(detail['models_with_materials']) + len(detail['models_without_materials'])
            coverage = (len(detail['models_with_materials']) / total * 100) if total > 0 else 0
            row = [
                detail['repo'],
                detail['num_models'],
                "; ".join(f"{k}:{v}" for k, v in detail['model_types'].items()),
                "; ".join(f"{k}:{v}" for k, v in detail.get('material_types', {}).items()),
                detail['has_material'],
                len(detail['models_with_materials']),
                len(detail['models_without_materials']),
                f"{coverage:.2f}",
                len(detail['invalid_files']),
                "; ".join(f"{k}:{v}" for k, v in detail.get('invalid_type_counts', {}).items())
            ]
            if write_model_paths:
                row.insert(5, " | ".join(detail['models_with_materials']))
                row.insert(6, " | ".join(detail['models_without_materials']))
            writer.writerow(row)

def print_summary(src, stats, verbose=True):
    print(f"\n[3D Model Analysis Summary for {src}]")
    print(f"  Total repos with models: {stats['n_repos']}")
    print(f"  Total zip files: {stats['n_zips']}")
    print(f"  Total 3D model files: {stats['n_models']}")
    print(f"  3D model files without materials: {stats['n_models_without_materials']}")
    print(f"  Repos with at least one material: {stats['repos_with_materials']}")
    print(f"  Repos with invalid files: {len(stats['repos_with_invalid_files'])}")
    print(f"  Total invalid files found: {stats['invalid_file_count']}")

    print(f"\n  Model type breakdown:")
    for ext, count in sorted(stats['model_type_counts'].items(), key=lambda x: -x[1]):
        print(f"    {ext}: {count}")

    print(f"\n  Material type breakdown:")
    for ext, count in sorted(stats['material_type_counts'].items(), key=lambda x: -x[1]):
        print(f"    {ext}: {count}")

    if stats["repos_with_invalid_files"]:
        if verbose:
            print(f"\n[Warning] Repos containing invalid files:")
            for repo_name, files in stats["repos_with_invalid_files"]:
                print(f"  - {repo_name}: {' | '.join(files)}")
        else:
            repo_names = [r[0] for r in stats["repos_with_invalid_files"]]
            preview = ', '.join(repo_names[:10])
            suffix = "..." if len(repo_names) > 10 else ""
            print(f"\n[Warning] Invalid files found in repos: {preview}{suffix}")

if __name__ == "__main__":
    root_in = '/labworking/Users/jyang/data/objaverse'
    srcs = {
        'github': 'github/repos',
    }
    exclude_invalid_repos = {'mattdeitke'}
    verbose = False
    multiprocess = True

    tik = time.time()
    for src, rel_path in srcs.items():
        path_in = os.path.join(root_in, rel_path)
        print(f'\n==== Analyzing source: {src.upper()} ====')
        stats = analyze_all_zips(path_in, exclude_invalid_repos=exclude_invalid_repos, use_multiprocess=multiprocess)
        print(f"Analysis completed for {src}. Found {stats['n_repos']} repos with models.")
        save_asset_texture_table(src, stats)
        print_summary(src, stats, verbose=verbose)
        save_reports(src, stats) # model_paths
    duration = time.time() - tik
    avg_time = duration / stats["n_models"] if stats.get("n_models", 0) else 0
    print(f"\nTotal analysis time: {time.strftime('%H:%M:%S', time.gmtime(duration))}")
    print(f"Averaged {avg_time:.2f} seconds per model.")
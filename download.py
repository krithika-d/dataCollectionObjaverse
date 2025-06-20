import sys
import os

# Tell Blender where to find pip-installed packages
sys.path.append("/Users/krithikadharanikota/blender-py-packages")

# DEBUG: Show all paths Blender is using
print("Python sys.path:")
for p in sys.path:
    print(p)

# Now import the module
import objaverse.xl as oxl

#import objaverse
import objaverse.xl as oxl
import multiprocessing
from typing import Dict, Hashable, Any

def handle_new_object(local_path: str, file_identifier: str, sha256: str, metadata: Dict[Hashable, Any]) -> None:
    print("\n📥 NEW OBJECT DOWNLOADED")
    print(f"  local_path={local_path}\n  file_identifier={file_identifier}\n  sha256={sha256}\n  metadata={metadata}")

def handle_missing_object(file_identifier: str, sha256: str, metadata: Dict[Hashable, Any]) -> None:
    print("\n⚠️ MISSING OBJECT")
    print(f"  file_identifier={file_identifier}\n  sha256={sha256}\n  metadata={metadata}")

def handle_modified_object(local_path: str, file_identifier: str, new_sha256: str, old_sha256: str, metadata: Dict[Hashable, Any]) -> None:
    print("\n🔁 MODIFIED OBJECT")
    print(f"  local_path={local_path}\n  file_identifier={file_identifier}\n  old_sha256={old_sha256}\n  new_sha256={new_sha256}\n  metadata={metadata}")

def handle_found_object(local_path: str, file_identifier: str, sha256: str, metadata: Dict[Hashable, Any]) -> None:
    print("\n✅ OBJECT ALREADY EXISTS")
    print(f"  local_path={local_path}\n  file_identifier={file_identifier}\n  sha256={sha256}\n  metadata={metadata}")

def main():
    # https://colab.research.google.com/drive/15XpZMjrHXuky0IgBbXcsUtb_0g-XWYmN?usp=sharing#scrollTo=MoauJlFiSQW-
    # https://github.com/TencentARC/InstantMesh/issues/165
    # download_dir = "/labworking/Users/jyang/data/objaverse"
    download_dir = "~/Desktop/hustle/ict-vgl/data/objaverse"

    # Step 0: remove previously downloaded objects
    # shutil.rmtree(os.path.expanduser(download_dir), ignore_errors=True)

    # Step 1: Download Annotations
    annotations = oxl.get_annotations(download_dir=download_dir)
    print(f'\n✅ Annotations downloaded to: {download_dir}')
    print(f'🔢 Source distribution:\n{annotations["source"].value_counts()}')
    print(f'🔢 FileType distribution:\n{annotations["fileType"].value_counts()}')
    print('📄 Sample annotations:')
    print(annotations.sample(5))

    # Step 2: Download Alignment Annotations
    alignment_annotations = oxl.get_alignment_annotations(download_dir=download_dir)
    print(f'\n✅ Alignment annotations also downloaded to: {download_dir}')

    # Step 3: Sample One Object Per Source
    n_sample = 5
    # annotations_filtered = annotations[annotations['source'] == 'smithsonian'] # Exclude 'smithsonian' source, since it's only has 2k objects
    # annotations_filtered = annotations[annotations['source'] == 'github'] # Exclude 'smithsonian' source, since it's only has 2k objects
    annotations_filtered = annotations[annotations['source'] == 'sketchfab'] # Exclude 'smithsonian' source, since it's only has 2k objects
    sampled_df = annotations_filtered.groupby('source').apply(lambda x: x.sample(n_sample)).reset_index(drop=True)
    print(f'\n📦 Sampled annotations ({n_sample} per source):\n{sampled_df}')

    # Step 4: Download the Sampled Objects
    # confirm = input("\nDo you want to download the sampled objects? (y/n): ").strip().lower()
    # if confirm != 'y':
    #     print("Download aborted.")
    #     return
    # oxl.download_objects(
    #     # Base parameters:
    #     objects: pd.DataFrame,
    #     download_dir: str = download_dir,
    #     # processes: Optional[int] = None,  # None => multiprocessing.cpu_count()
    #     processes: Optional[int] = multiprocessing.cpu_count() / 2,  # None => multiprocessing.cpu_count()

    #     # optional callback functions:
    #     handle_found_object: Optional[Callable] = None,
    #     handle_modified_object: Optional[Callable] = None,
    #     handle_missing_object: Optional[Callable] = None,

    #     # GitHub specific:
    #     save_repo_format: Optional[Literal["zip", "tar", "tar.gz", "files"]] = None,
    #     handle_new_object: Optional[Callable] = None,
    # )
    oxl.download_objects(
        objects=sampled_df,
        download_dir=download_dir,
        processes=int(64),
        handle_new_object=handle_new_object,
        handle_missing_object=handle_missing_object,
        handle_modified_object=handle_modified_object,
        save_repo_format="zip",
        handle_found_object=handle_found_object,
    )

    # clean up tmp via the following commands. 
    # find /tmp -maxdepth 1 -type d -user $USER # preview what would be deleted first
    # find /tmp -maxdepth 1 -type d -user $USER -exec rm -rf {} + # delete

    # TODO: add data check, need to loop through all downloaded data


if __name__ == "__main__":
    main()

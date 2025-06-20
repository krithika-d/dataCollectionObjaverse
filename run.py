import sys
sys.path.append("/Users/krithikadharanikota/blender-py-packages")
import subprocess
import json
import trimesh
import os



#os.environ['CUDA_VISIBLE_DEVICES'] = '4'

def run(frame_id, l, r, t, mesh_path, out_path):
    subprocess.run([
        #'/scratch/jiyang/projects/blender/blender-4.1.1-linux-x64/blender', 
        '/Applications/Blender.app/Contents/MacOS/Blender',
        '-b', '-noaudio', '-P',
        './blender_shadow.py',
        '--',
        f'{frame_id}',
        f'{l[0]}', f'{l[1]}', f'{l[2]}',
        f'{r[0]}', f'{r[1]}', f'{r[2]}', f'{r[3]}', f'{r[4]}', f'{r[5]}', f'{r[6]}', f'{r[7]}', f'{r[8]}',
        f'{t[0]}', f'{t[1]}', f'{t[2]}',
        f'{mesh_path}',
        f'{out_path}',
    ])
    
if __name__ == '__main__':
    
    n_li = 6
    # f_path_sfm = '/dlbimg/data/volumetric/cache/ID_01245_XercesBlue_3K_r2020-Linear_G24int_Full_062023/00000001/StructureFromMotion/f50ed9695263cb54cfcbf1a8ab129c91bf4aeb52/cameras.sfm'
    # f_path_mesh = '/dlbimg/data/volumetric/cache/ID_01245_XercesBlue_3K_r2020-Linear_G24int_Full_062023/00000001/Meshing/13149d66bfc766196f7a7cbf1ef78612a7e89af0/mesh.obj'
    # f_path_out = f'/dlbimg/projects/relightable_volumetric/logs/cycle_00000001_Li_{n_li}'
    
    # f_path_sfm = '/scratch/jiyang/projects/relightable_volumetric/data/volumetric/cache/ID_01245_XercesBlue_3K_r2020-Linear_G24int_Full_062023/00000001/StructureFromMotion/63433a3dd22e16323add09c5df81a33e57857936/cameras.sfm'
    # f_path_mesh = '/scratch/jiyang/projects/relightable_volumetric/data/volumetric/cache/ID_01245_XercesBlue_3K_r2020-Linear_G24int_Full_062023/00000001/Texturing/a336eed124af61bdfa2db8deb5eab5c9fb8ed2f7/texturedMesh.obj'
    # f_path_pcl = '/scratch/jiyang/projects/relightable_volumetric/data/volumetric/cache/ID_01245_XercesBlue_3K_r2020-Linear_G24int_Full_062023/00000001/Texturing/a336eed124af61bdfa2db8deb5eab5c9fb8ed2f7/texturedPointCloud.ply'
    # f_path_out = f'/scratch/jiyang/projects/relightable_volumetric/data/blender/cycle_00001000_Li_{n_li}'
    
    f_path_sfm = "/Users/krithikadharanikota/Desktop/hustle/ict-vgl/data/dummy_cameras.sfm"      
    f_path_mesh = "/Users/krithikadharanikota/Desktop/hustle/ict-vgl/data/objaverse/hf-objaverse-v1/glbs/000-023/741950dc0247450b87c952b625aa239b.glb"        # .glb or .obj
    f_path_out  = "/Users/krithikadharanikota/Desktop/ict-vgl-renders"


    # 
    
    # load sfm data as text and convert to json
    with open(f_path_sfm, 'r') as f:
        sfm = json.load(f)
        
    # convert a list of object to a dictionary by 'poseID' as key and the 'pose' as value
    pose_dict = {pose['poseId']: pose for pose in sfm['poses']}
    
    if n_li == 6:
        # generate lighting position from x, y, z, -x, -y, -z
        Ls = [
            [0, 0, 10],
            # [0, 0, -10],
            # [0, 10, 0],
            # [0, -10, 0],
            # [10, 0, 0],
            # [-10, 0, 0],
        ]
    elif n_li == 162:
        # uniformly sample the lighting position on a sphere
        geometry = trimesh.creation.icosphere(subdivisions=2, radius=10)
        Ls = geometry.vertices
    else:
        raise ValueError(f'Invalid number of lighting positions: {n_li}')
    
    for l in Ls:
        for view in sfm['views']:
            
            print(f'set light {l} with view {view["frameId"]}')
            
            frame_id = view['frameId']
            pose_id = view['poseId']
            
            pose = pose_dict[pose_id]['pose']['transform']
            r = pose['rotation']
            t = pose['center']
            
            run(frame_id, l, r, t, f_path_mesh, f_path_out)
import numpy as np 
import open3d as o3d 
import os 
from tqdm import tqdm
import argparse


def pcd2np(path):
    
    pcd = o3d.t.io.read_point_cloud(path)
    ## 
    xyz = pcd.point['positions'].numpy()
    if 'normals' in pcd.point:
        normals = pcd.point['normals'].numpy()
        pts = np.concatenate((xyz, normals[:, 1][:, np.newaxis]), axis = 1)
    if 'intensity' in pcd.point: 
        intensity = pcd.point['intensity'].numpy()
        pts = np.concatenate((xyz, intensity), axis = 1)

    return pts

def saveasbin(pcd, outpath): 
    # remove nan
    pcd = pcd[~np.isnan(pcd[:, 0])]
    # remove 0 values
    pcd = pcd[(pcd[:, 0] != 0)]
    with open(outpath, 'w') as f:
        pcd.astype(np.float32).tofile(f)

def convert(dataroot, out):
    files = os.listdir(dataroot)

    for file in tqdm(files): 
        if file[-3:] != 'pcd': 
            continue 
        path = os.path.join(dataroot, file)
      
        pts = pcd2np(path)
        outpath = os.path.join(out, file[:-3] + 'bin')
        saveasbin(pts, outpath)


if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument(
        "--dataroot", type=str, default=None, help="indicate dataroot of the folder"
    )
    parser.add_argument(
        "--out", type=str, default=None, help="indicate result folder"
    )
    args = parser.parse_args()

    dataroot = args.dataroot
    outfolder = args.out
    if not os.path.exists(outfolder): 
        os.makedirs(outfolder)

    convert(dataroot, outfolder)
    

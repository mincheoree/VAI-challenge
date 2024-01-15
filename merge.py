import os 
from tqdm import tqdm
import open3d as o3d 
import numpy as np 

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

if __name__ == '__main__':

    rootpath = 'data/custom/yeouido/'
    out = 'data/custom/yeouido_merged'
    folders = os.listdir(rootpath)
    folders = folders[16:]
    files = []
    if not os.path.exists(out): 
            os.makedirs(out)
    for folder in folders: 
        outpath = os.path.join(out, folder)
        if not os.path.exists(outpath): 
            os.makedirs(outpath)
        
        path = os.path.join(rootpath, folder, 'plog')
        files = os.listdir(path)
        for i, file in enumerate(files):
            inputpath = os.path.join(path, file)
        
            pts = pcd2np(inputpath)
            savepath = os.path.join(outpath, "{:06d}.bin".format(i))
            saveasbin(pts, savepath)
  

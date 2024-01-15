import open3d as o3d
import numpy as np 
from tools.visual_utils import open3d_vis_utils as V

def adjust_pitch(pts): 
    depth = np.linalg.norm(pts, 2, axis=1)
    scan_x = pts[:, 0]
    scan_y = pts[:, 1]
    scan_z = pts[:, 2]
    
    # get angles of all points
    yaw = -np.arctan2(scan_y, scan_x)
    pitch = np.arcsin(scan_z / depth)
    pitch -= (10 / 180 * np.pi)
    xy = np.linalg.norm(pts[:, :2], 2, axis = 1)
    depth = xy / np.cos(pitch) 
    new_x = depth * np.cos(pitch) * np.cos(yaw)
    new_y = (-1) * depth * np.cos(pitch) * np.sin(yaw)
    new_z = depth * np.sin(pitch)
    new_pts = np.stack([new_x, new_y, new_z], axis = 1)
    return new_pts

def adjust_z(pts): 
    scan_z = pts[:, 2]
    deltaz = 0.22*pts[:, 0]
    scan_z += deltaz
    pts[:, 2] = scan_z 
    return pts

# path = './data/custom/yeouido/231207-142408_SMPL00007_R192p125h_STRAIGHT_PATROL_center/plog/frm_00000001.pcd'
# pts = pcd2np(path)

# pts = np.fromfile('./data/custom/patrol_merged/000000.bin', dtype = np.float32).reshape(-1, 4)
points = np.load('tools/points.npy')
boxes = np.load('tools/gt_boxes.npy')
labels = np.ones(boxes.shape[0], dtype = np.int)
V.draw_scenes(points = points,  ref_boxes =  boxes, ref_labels = labels)

# o3d.visualization.draw_geometries([newpcd, coords])

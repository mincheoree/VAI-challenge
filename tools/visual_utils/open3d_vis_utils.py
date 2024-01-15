"""
Open3d visualization tool box
Written by Jihan YANG
All rights preserved from 2021 - present.
"""
import open3d
import torch
import matplotlib
import numpy as np
import json
import pickle

# box_colormap = [
#     [1, 1, 1],
#     [0, 1, 0],
#     [0, 1, 1],
#     [1, 1, 0],
# ]

box_colormap = [
    [255, 158, 0],
    [112, 113, 232],
    [198, 131, 215],
    [237, 158, 214],
   [255, 199, 199],
    [112, 128, 144],
   [154,222, 123],
    [80, 141, 105],
     [0, 0, 230],
    [47, 79, 79],
]

def get_coor_colors(obj_labels):
    """
    Args:
        obj_labels: 1 is ground, labels > 1 indicates different instance cluster

    Returns:
        rgb: [N, 3]. color for each point.
    """
    colors = matplotlib.colors.XKCD_COLORS.values()
    max_color_num = obj_labels.max()

    color_list = list(colors)[:max_color_num+1]
    colors_rgba = [matplotlib.colors.to_rgba_array(color) for color in color_list]
    label_rgba = np.array(colors_rgba)[obj_labels]
    label_rgba = label_rgba.squeeze()[:, :3]

    return label_rgba


def draw_scenes(points, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None, point_colors=None, draw_origin=True):
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy()
    if isinstance(ref_boxes, torch.Tensor):
        ref_boxes = ref_boxes.cpu().numpy()

    vis = open3d.visualization.Visualizer()
    vis.create_window()

    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.zeros(3)

    # draw origin
    if draw_origin:
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)

    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])

    vis.add_geometry(pts)
    if point_colors is None:
        pts.colors = open3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))
    else:
        pts.colors = open3d.utility.Vector3dVector(point_colors)

    if gt_boxes is not None:
        vis = draw_box(vis, gt_boxes, (0, 0, 1))

    if ref_boxes is not None:
        vis = draw_box(vis, ref_boxes, (0, 1, 0), ref_labels, ref_scores)

    vis.run()
    vis.destroy_window()


def translate_boxes_to_open3d_instance(gt_boxes):
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """
    center = gt_boxes[0:3]
    lwh = gt_boxes[3:6]
    axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
    rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)

    line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    # import ipdb; ipdb.set_trace(context=20)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = open3d.utility.Vector2iVector(lines)

    return line_set, box3d


def draw_box(vis, gt_boxes, color=(0, 1, 0), ref_labels=None, score=None):
    for i in range(gt_boxes.shape[0]):
        line_set, box3d = translate_boxes_to_open3d_instance(gt_boxes[i])
        if ref_labels is None:
            line_set.paint_uniform_color(color)
        else:
            line_set.paint_uniform_color(np.array(box_colormap[ref_labels[i]-1])/255)

        vis.add_geometry(line_set)

        # if score is not None:
        #     corners = box3d.get_box_points()
        #     vis.add_3d_label(corners[5], '%.2f' % score[i])
    return vis

def get_sweep(sweep_info, root_path):
        def remove_ego_points(points, center_radius=1.0):
            mask = ~((np.abs(points[:, 0]) < center_radius) & (np.abs(points[:, 1]) < center_radius))
            return points[mask]

        lidar_path = root_path + sweep_info['lidar_path']
        points_sweep = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])[:, :4]
        points_sweep = remove_ego_points(points_sweep).T
        if sweep_info['transform_matrix'] is not None:
            num_points = points_sweep.shape[1]
            points_sweep[:3, :] = sweep_info['transform_matrix'].dot(
                np.vstack((points_sweep[:3, :], np.ones(num_points))))[:3, :]

        cur_times = sweep_info['time_lag'] * np.ones((1, points_sweep.shape[1]))
        return points_sweep.T, cur_times.T

def get_lidar_with_sweeps(info, max_sweeps=9):
        root_path = 'data/nuscenes/v1.0-trainval/'
        lidar_path =  root_path + info['lidar_path']
        points = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])[:, :4]
        
        sweep_points_list = [points]
        sweep_times_list = [np.zeros((points.shape[0], 1))]

        for k in np.random.choice(len(info['sweeps']), max_sweeps - 1, replace=False):
            points_sweep, times_sweep = get_sweep(info['sweeps'][k], root_path)
            sweep_points_list.append(points_sweep)
            sweep_times_list.append(times_sweep)

        points = np.concatenate(sweep_points_list, axis=0)
        times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)

        points = np.concatenate((points, times), axis=1)
        return points

def read_json(input_path):
    with open(input_path, 'r') as j:
        abc = json.loads(j.read())['results']
    return abc 

if __name__ == '__main__': 
    base_path = 'output/nuscenes_models/centerpoint_v2/default/eval/epoch_6898/val/default/final_result/data/results_nusc.json'
    ours_path = 'output/nuscenes_models/centerpoint_v2/default/eval/epoch_6941/val/default/final_result/data/results_nusc.json'
    infos = pickle.load(open('data/nuscenes/v1.0-trainval/nuscenes_infos_10sweeps_val.pkl', 'rb'))
    base, ours = read_json(base_path), read_json(ours_path)
    binfo = base['501491486c9e46d6b30d01c7db5b4f32']
    oinfo = ours['501491486c9e46d6b30d01c7db5b4f32']
    temp = list(base.items())
    idx = [idx for idx, key in enumerate(temp) if key[0] == '501491486c9e46d6b30d01c7db5b4f32'][0]
    base_preds = pickle.load(open('output/nuscenes_models/centerpoint_v2/default/eval/epoch_6898/val/default/result.pkl', 'rb'))
    our_preds = pickle.load(open('output/nuscenes_models/centerpoint_v2/default/eval/epoch_6941/val/default/result.pkl', 'rb'))
    base_pred = base_preds[idx]
    our_pred = our_preds[idx]
    info = infos[idx]
    points = get_lidar_with_sweeps(info)
    gt_boxes = info['gt_boxes']
    base_pred_boxes = base_pred['boxes_lidar']
    base_pred_labels = base_pred['pred_labels'] 
    our_pred_boxes = our_pred['boxes_lidar']
    our_pred_labels = our_pred['pred_labels'] 
    
    np.save('base/gtbox', gt_boxes)
    np.save('base/gtlabels', info['gt_names'])
    np.save('base/predbox', base_pred_boxes)
    np.save('base/predlabel', base_pred_labels)
    np.save('base/points', points)
    np.save('ours/predbox', our_pred_boxes)
    np.save('ours/predlabel', our_pred_labels)


    # draw_scenes(points, gt_boxes, pred_boxes)

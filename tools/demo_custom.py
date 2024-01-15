import _init_path
import argparse
import glob
from pathlib import Path

try:
    import open3d
    from visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V
    OPEN3D_FLAG = False

import numpy as np
import torch
import os

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from pyquaternion import Quaternion

def boxes_to_corners_3d(boxes3d):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    Returns:
    """
    boxes3d, is_numpy = common_utils.check_numpy_to_torch(boxes3d)

    template = boxes3d.new_tensor((
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
    )) / 2

    corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
    corners3d = common_utils.rotate_points_along_z(corners3d.view(-1, 8, 3), boxes3d[:, 6]).view(-1, 8, 3)
    corners3d += boxes3d[:, None, 0:3]

    return corners3d.numpy() if is_numpy else corners3d

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
    # deltaz = 0.22* np.linalg.norm(pts[:, :2], 2, axis = 1)
    deltaz = 0.22* pts[:, 0]

    scan_z -= deltaz
    pts[:, 2] = scan_z 
    return pts

def rotate(points): 
        angle = 10
        # rotate along y axis
        points = points[:, :3] = np.dot(
            Quaternion(
                axis=[0, 1, 0],
                # adjust angle in radian
                radians=  angle / 180 * np.pi,
            ).rotation_matrix,
            points[:, :3].T,
        ).T

        return points

class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.custom_infos = []
        # custom_infos = os.listdir(os.path.join(self.root_path, 'patrol2'))
        custom_infos = os.listdir(self.root_path)
        self.custom_infos.extend(custom_infos)
       
    def __len__(self):
        return len(self.custom_infos)

    def get_lidar(self, idx):
        # lidar_file = self.root_split_path / 'velodyne' / ('%s.bin' % idx)
        # lidar_file = self.root_path / 'patrol2' / self.custom_infos[idx]
        lidar_file = self.root_path  / self.custom_infos[idx]
        
        assert lidar_file.exists()
        channels = [0, 1, 2, 4]
        return np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4) 
        #return np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 6)[:, channels] 

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.custom_infos)

        # info = copy.deepcopy(self.custom_infos[index])

        points = self.get_lidar(index)
        # for nuscenes 
        # points = np.concatenate((points, np.zeros((points.shape[0], 1))), axis = 1)
        # for kitti, normalize intensity to 0-1
        points[:, -1] = points[:, -1]/255.0
        
        
        points[:, :3] = np.dot(
                Quaternion(
                    axis=[0, 0, 1],
                    radians= - 90 / 180 * np.pi,
                ).rotation_matrix,
                points[:, :3].T,
            ).T
        # 10 degrees rotate along y
        points[:, :3] = rotate(points)
        # 1m z axis up
        points[:, 2] += 1
        # points[:, :3] = adjust_z(points[:, :3])
        
        input_dict = {
            'frame_id': self.custom_infos[index].split('/')[-1],
            'points': points
        }
       

        data_dict = self.prepare_data(data_dict=input_dict)

        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)
            
            V.draw_scenes(
                points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
                ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
            )

            if not OPEN3D_FLAG:
                mlab.show(stop=True)

    logger.info('Demo done.')


if __name__ == '__main__':
    main()

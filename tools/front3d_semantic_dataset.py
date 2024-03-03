
from typing import Any, Callable, List, Optional, Tuple, Union, Dict

import os
import random
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch import nn, Tensor, normal
from tqdm import tqdm
from abc import ABC
import time
import open3d as o3d

#give me 19 different colors mapping from id to color

def get_color_map():
    color_map = [
        [0, 0, 0],  # 0
        [0, 0, 128],  # 1
        [0, 128, 0],  # 2
        [0, 128, 128],  # 3
        [128, 0, 0],  # 4
        [128, 0, 128],  # 5
        [128, 128, 0],  # 6
        [128, 128, 128],  # 7
        [0, 0, 64],  # 8
        [0, 0, 192],  # 9
        [0, 128, 64],  # 10
        [0, 128, 192],  # 11
        [128, 0, 64],  # 12
        [128, 0, 192],  # 13
        [128, 128, 64],  # 14
        [128, 128, 192],  # 15
        [0, 64, 0],  # 16
        [0, 64, 128],  # 17
        [0, 192, 0],  # 18
        [0, 192, 128],  # 19
        [128, 64, 0],  # 20
    ]

    return color_map

class BaseDataset(torch.utils.data.Dataset, ABC):
    """Base dataset class"""

    def __init__(
        self,
        dataset_type: str = None,
        features_path: str = None,
        boxes_path: str = None,
        out_feat_path: str = None,
        sem_feat_path: str = None,
        scene_list: Optional[List[str]] = None,
        normalize_density: bool = True,
        flip_prob: float = 0.0,
        rotate_prob: float = 0.0,
        rot_scale_prob: float = 0.0,
        z_up: bool = True,
    ) -> None:
        super().__init__()
        self.dataset_type = dataset_type  # hypersim, 3dfront, general
        self.features_path = features_path
        self.boxes_path = boxes_path
        self.out_feat_path = out_feat_path
        self.sem_feat_path = sem_feat_path
        self.scene_list = scene_list
        self.normalize_density = normalize_density
        self.flip_prob = (
            flip_prob  # the probability of flipping the data along the horizontal axes
        )
        self.rotate_prob = (
            rotate_prob  # the probability of rotating the data by 90 degrees
        )
        self.rot_scale_prob = (
            rot_scale_prob  # the probability of extra rotation and scaling
        )
        self.z_up = z_up  # whether the boxes are z-up

        self.scene_data = []

    def construct_grid(self, res):
        res_x, res_y, res_z = res
        
        # Create 1D tensors for x, y, z
        x = torch.linspace(0, res_x, res_x, dtype=torch.float32)
        y = torch.linspace(0, res_y, res_y, dtype=torch.float32)
        z = torch.linspace(0, res_z, res_z, dtype=torch.float32)

        # Use broadcasting to scale all tensors simultaneously
        scale = torch.max(torch.tensor(res, dtype=torch.float32))
        x = x / scale + 0.5 / scale
        y = y / scale + 0.5 / scale
        z = z / scale + 0.5 / scale

        # Create grids using broadcasting
        X, Y, Z = torch.meshgrid(x, y, z)

        # Reshape and concatenate to get the final grid
        grid = torch.stack((X.reshape(-1), Y.reshape(-1), Z.reshape(-1)), dim=1)

        # print("grid shape", grid.shape)
        return grid

    def load_single_scene(self, scene: str):
        """
        Load a single scene
        """
        if self.boxes_path is None:
            boxes = None
        else:
            boxes = torch.from_numpy(
                np.load(os.path.join(self.boxes_path, scene + ".npy"))
            )

        if self.out_feat_path is None:
            out_rgbsigma = None
        else:
            out_features_path = os.path.join(self.out_feat_path, scene + ".npz")
            with np.load(out_features_path) as features:
                out_rgbsigma = features["rgbsigma"]
                if self.normalize_density:
                    out_alpha = self.density_to_alpha(out_rgbsigma[..., -1])
                    out_rgbsigma[..., -1] = out_alpha

                # From (W, L, H, C) to (C, W, L, H)
                out_rgbsigma = np.transpose(out_rgbsigma, (3, 0, 1, 2))
                out_rgbsigma = torch.from_numpy(out_rgbsigma)

                if out_rgbsigma.dtype == torch.uint8:
                    # normalize rgbsigma to [0, 1]
                    out_rgbsigma = out_rgbsigma.float() / 255.0

        if self.sem_feat_path is None:
            out_sem = None
        else:
            out_sem_path = os.path.join(self.sem_feat_path, scene + ".npy")
            out_sem = np.load(out_sem_path)
            out_sem = torch.from_numpy(out_sem)
            out_sem = out_sem.reshape(-1, 1)


        scene_features_path = os.path.join(self.features_path, scene + ".npz")
        # try:

        # load_start = time.time()
        with np.load(scene_features_path) as features:
            rgbsigma = features["rgbsigma"]
            alpha = self.density_to_alpha(rgbsigma[..., -1])
            rgbsigma[..., -1] = alpha
            res = features["resolution"]
        # print("load time", time.time() - load_start)
        # rgbsigma = np.transpose(rgbsigma, (2, 1, 0, 3)).reshape(-1, 4)
        
        rgbsigma = rgbsigma.reshape(-1, 4)
        alpha = rgbsigma[:, -1]
        mask = alpha > 0.01
        
        # res = res[[2, 0, 1]]
        grid = self.construct_grid(res)

        point = grid[mask, :]

        # rgbsigma = rgbsigma[:,:3][mask, :]
        if self.sem_feat_path is not None:
            out_sem = out_sem[mask]

        #Now subsample point cloud to keep 20k points

        if point.shape[0] > 50000:
            idx = np.random.choice(point.shape[0], 50000, replace=False)
            point = point[idx, :]
            if self.sem_feat_path is not None:
                out_sem = out_sem[idx]

        elif point.shape[0] < 50000:
            idx = np.random.choice(point.shape[0], 50000 - point.shape[0], replace=True)
            point = torch.cat([point, point[idx, :]], 0)
            #point = np.concatenate([point, point[idx, :]], 0)
            # if self.sem_feat_path is not None:
            #     out_sem = np.concatenate([out_sem, out_sem[idx]], 0)

        # if rgbsigma.dtype == torch.uint8:
        #     # normalize rgbsigma to [0, 1]
        #     rgbsigma = rgbsigma.float() / 255.0

        # point = np.concatenate([point, rgbsigma], 1)


        # scene_features_path = os.path.join(self.features_path, scene + ".npz")
        # with np.load(scene_features_path) as features:
        #     rgbsigma = features["rgbsigma"]
        #     if self.normalize_density:
        #         alpha = self.density_to_alpha(rgbsigma[..., -1])
        #         rgbsigma[..., -1] = alpha

        #     # From (W, L, H, C) to (C, W, L, H)
        #     rgbsigma = np.transpose(rgbsigma, (3, 0, 1, 2))
        #     rgbsigma = torch.from_numpy(rgbsigma)

        #     if rgbsigma.dtype == torch.uint8:
        #         # normalize rgbsigma to [0, 1]
        #         rgbsigma = rgbsigma.float() / 255.0

        # if self.out_feat_path is not None:
        #     return scene, rgbsigma, out_rgbsigma
        if self.sem_feat_path is not None:
            return point, out_sem
        else:
            return point, scene

    def load_scene_data(self, preload: bool = False, percent_train=1.0):
        """
        Check scene data and load them if needed
        """
        # if self.scene_list is None:
        #     # if scene_list is not provided, use all scenes in feature path
        #     feature_names = os.listdir(self.features_path)
        #     self.scene_list = [
        #         f.split(".")[0] for f in feature_names if f.endswith(".npz")
        #     ]

        # print("==================================================\n\n\n")
        # print("self.scene_list", len(self.scene_list))

        # print("percent_train", percent_train)
        # print("len(self.scene_list)", len(self.scene_list))
        # print("==================================================\n\n\n")
        num_train = int(percent_train * len(self.scene_list))
        self.scene_list = self.scene_list[:num_train]

        scenes_kept = []
        for scene in tqdm(self.scene_list):
            scene_features_path = os.path.join(self.features_path, scene + ".npz")
            if not os.path.isfile(scene_features_path):
                print(f"{scene} does not have a feature file")
                continue
            if self.boxes_path is not None:
                boxes = np.load(os.path.join(self.boxes_path, scene + ".npy"))
                if boxes.shape[0] == 0:
                    print(f"{scene} does not have any boxes")
                    continue

            scenes_kept.append(scene)

        self.scene_list = scenes_kept
        if preload:
            self.scene_data = [
                self.load_single_scene(scene) for scene in self.scene_list
            ]

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor]:
        if self.scene_data:
            point, out_sem = self.scene_data[index]
        else:
            scene = self.scene_list[index]
            point, out_sem = self.load_single_scene(scene)

        return point, out_sem

    def __len__(self) -> int:
        return len(self.scene_list)

    @staticmethod
    def density_to_alpha(density):
        return np.clip(1.0 - np.exp(-np.exp(density) / 100.0), 0.0, 1.0)

    @staticmethod
    def collate_fn(
        batch: List[Tuple[Tensor, Tensor, Tensor]]
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        # The network expects the features and boxes of different scenes to be in two lists
        rgbsigma = []
        boxes = []
        scenes = []
        for sample in batch:
            rgbsigma.append(sample[0])
            boxes.append(sample[1])
            # scenes.append(sample[2])
        return rgbsigma, boxes
    
class Front3DSemanticDataset(BaseDataset):
    def __init__(
        self,
        features_path: str,
        sem_feat_path: str,
        scene_list: Optional[List[str]] = None,
        normalize_density: bool = True,
        flip_prob: float = 0.0,
        rotate_prob: float = 0.0,
        rot_scale_prob: float = 0.0,
        preload: bool = False,
        percent_train=1.0,
    ):
        super().__init__(
            "3dfront",
            features_path,
            None,
            None,
            sem_feat_path,
            scene_list,
            normalize_density,
            flip_prob,
            rotate_prob,
            rot_scale_prob,
        )
        self.load_scene_data(preload=preload, percent_train=percent_train)


if __name__ == '__main__':

    dataset_split = '/home/zubairirshad/Downloads/front3d_rpn_data/front3d_split.npz'
    with np.load(dataset_split) as split:
        train_scenes = split["train_scenes"]
        test_scenes = split["test_scenes"]
        val_scenes = split["val_scenes"]

    features_path = '/home/zubairirshad/Downloads/front3d_rpn_data/features'
    sem_feat_path = '/home/zubairirshad/Downloads/front3d_rpn_data/voxel_front3d'
    dataset = Front3DSemanticDataset(
        features_path=features_path,
        sem_feat_path=sem_feat_path,
        scene_list=train_scenes,
        preload=False,
        percent_train=1.0,
    )

    # trainDataLoader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True, num_workers=10, drop_last=True)

    trainDataLoader =  torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        collate_fn=dataset.collate_fn,
    )

    for i, data in enumerate(trainDataLoader):
        point, alpha, out_sem = data

        # point = point[0]
        # alpha = alpha[0]
        # out_sem = out_sem[0]

        # print("np.unique(out_sem)", np.unique(out_sem))


        if torch.cuda.is_available():
            point = torch.stack([item.cuda() for item in point])
            out_sem = torch.stack([item.cuda() for item in out_sem])
            # alpha = [item.cuda() for item in alpha]
            # out_sem = [item.cuda() for item in out_sem]
            # grid = [item.cuda() for item in grid]

        



        # mask = alpha > 0.01
        # point = grid[mask, :]
        # out_sem = out_sem[mask]

        print("point", point.shape)
        print("out_sem", out_sem.shape)
        break

    print("len(dataset)", len(dataset))

    # data = dataset[0]

    # point, alpha, out_sem = data

    # print("point", point.shape)
    # print("out_sem", out_sem.shape)

    # #Now let's visualize the point cloud and color them according to the semantic labels

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(point)
    # colors = get_color_map()
    # # out_sem = out_sem.squeeze(0)
    # print("out_sem", out_sem.shape)
    # semantic_colors = [np.array(colors[i])/255.0 for i in out_sem]
    # pcd.colors = o3d.utility.Vector3dVector(semantic_colors)
    # o3d.visualization.draw_geometries([pcd])


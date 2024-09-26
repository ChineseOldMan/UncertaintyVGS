#!/usr/bin/env python3
import numpy as np
import trimesh
import torch

from submodules.mast3r.mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from submodules.mast3r.dust3r.dust3r.image_pairs import make_pairs
# from utils.mast3r_image import load_images
from submodules.mast3r.dust3r.dust3r.utils.device import to_numpy
from submodules.mast3r.mast3r.demo import get_args_parser
from submodules.mast3r.mast3r.model import AsymmetricMASt3R
from submodules.mast3r.dust3r.dust3r.utils.image import _resize_pil_image
from utils.mast3r_utils import save_colmap_cameras, save_colmap_images, storePly, load_images
from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary, qvec2rotmat
import copy
import shutil
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def estimate_transform(source_P, target_P):
    A = source_P[:, :3, 3]
    B = target_P[:, :3, 3]
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    A_centered = A - centroid_A
    B_centered = B - centroid_B
    H = np.dot(A_centered.T, B_centered)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)
    t = centroid_B - np.dot(R, centroid_A)

    return R, t

def transform_poses(poses, R, t):
    transformed_poses = np.zeros_like(poses)
    for i in range(poses.shape[0]):
        current_pose = poses[i]
        transformed_R = np.dot(R, current_pose[:3, :3])
        transformed_T = np.dot(R, current_pose[:3, 3]) + t
        transformed_poses[i, :3, :3] = transformed_R
        transformed_poses[i, :3, 3] = transformed_T
        transformed_poses[i, 3, 3] = 1  # 齐次坐标的最后一行
        
    return transformed_poses

def transform_pcl(point_clouds, R, t):
    n = point_clouds.shape[0]
    ones = np.ones((n, 1))
    homogeneous_points = np.hstack((point_clouds, ones))  # 将点云转换为齐次坐标
    transformed_homogeneous_points = homogeneous_points @ np.vstack((R, t))  # 矩阵乘法
    return transformed_homogeneous_points[:, :3]

class SparseGAState():
    def __init__(self, sparse_ga, cache_dir=None, outfile_name=None):
        self.sparse_ga = sparse_ga
        self.cache_dir = cache_dir

    def __del__(self):

        if self.cache_dir is not None and os.path.isdir(self.cache_dir):
            shutil.rmtree(self.cache_dir)
        self.cache_dir = None

def reconstruct_scene(filepath, image_size, model, device, 
                      optim_params, cache_dir, output_colmap_path,
                      n_views, know_camera=True, shared_intrinsics=True, min_conf_thr=2):
    """
    Perform 3D reconstruction and return point cloud, camera intrinsics, and poses.
    
    Args:
        filelist (list): List of image file paths
        image_size (int): Size to resize images to
        model: The MASt3R model
        device: Computation device (e.g., 'cuda' or 'cpu')
        optim_params (dict): Optimization parameters
    
    Returns:
        tuple: (point_cloud, camera_intrinsics, camera_poses)
    """
    print("filepath", filepath)
    train_img_list = sorted(os.listdir(os.path.join(filepath, "images")))
    if args.llffhold > 0:
        train_img_list = [c for idx, c in enumerate(train_img_list) if (idx+1) % args.llffhold != 0]
    
    # sample sparse view
    indices = np.linspace(0, len(train_img_list) - 1, n_views, dtype=int)
    print(indices)
    tmp_img_list = [train_img_list[i] for i in indices]
    train_img_list = tmp_img_list
    # imgs = _resize_pil_image(train_img_list, image_size)
    print("train_img_list", train_img_list)
    # 在train_img_list中的每个图片路径前面拼接上filepath
    train_img_list = [os.path.join(filepath, "images", c) for c in train_img_list]
    imgs, ori_size = load_images(train_img_list, size=image_size)
    print("ori_size", ori_size)
    
    # Create image pairs
    pairs = make_pairs(imgs, scene_graph="complete", prefilter=None, symmetrize=True)
    
    # Perform sparse global alignment 
    # Pay attention to the cache_dir parameter! TODO
    scene = sparse_global_alignment(train_img_list, pairs, cache_path=cache_dir,
                                    model=model, device=device, shared_intrinsics=shared_intrinsics,
                                    **optim_params)
    scene_state = SparseGAState(scene, cache_dir)
    scene_ = scene_state.sparse_ga
    imgs = to_numpy(scene_.imgs)
    # # Extract point cloud
    pts3d, _, confs = scene_.get_dense_pts3d(clean_depth=True)
    if know_camera:
        # 内参 TODO 用于未知相机位姿时
        # focals = scene_.get_focals()
        # principal_points = scene_.get_principal_points()  # 获取主点
        # 构建相机内参矩阵 K
        # camera_intrinsics = torch.stack([
        #     focals, torch.zeros_like(focals), principal_points[:, 0],
        #     torch.zeros_like(focals), focals, principal_points[:, 1],
        #     torch.zeros_like(focals), torch.zeros_like(focals), torch.ones_like(focals)
        # ]).reshape(-1, 3, 3)
        # camera_intrinsics = to_numpy(camera_intrinsics)
        
        camera_poses = to_numpy(scene_.get_im_poses())
        msk = to_numpy([c > min_conf_thr for c in confs])
    
        # trasform to the right
        # 生成一个字符串“n_views”，其中n的值为n_views，后面的“_views”保留
        output_colmap_path = os.path.join(output_colmap_path, f"{n_views}_views")
        cameras_intrinsic_file = os.path.join(filepath, f"{n_views}_views", "colmap", "trangulated", "cameras.bin")
        cameras_extrinsic_file = os.path.join(filepath, f"{n_views}_views", "colmap", "trangulated", "images.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
            
        camera_poses_est = np.zeros((n_views, 4, 4))
        camera_poses_ori = np.zeros((n_views, 4, 4))
        for i in range(n_views):
            camera_poses_est[i] = np.linalg.inv(camera_poses)
            for j in range(len(cam_extrinsics)):
                if train_img_list[i-1][-3:] == cam_extrinsics[j].name[-3:]:
                    R = np.transpose(qvec2rotmat(cam_extrinsics[j].qvec))
                    T = np.array(cam_extrinsics[j].tvec)
                    camera_poses_ori[i, :3, :3] = R
                    camera_poses_ori[i, :3, 3] = T
                    break

        R_e2o, t_e2o = estimate_transform(camera_poses_est, camera_poses_ori)
        
        # camera_poses = transform_poses(camera_poses, R_e2o, t_e2o) # TODO 用于未知相机位姿时
        camera_poses = camera_poses_ori
    
        target_directory = os.path.join(filepath, f"{n_views}_views", "mast3r")
        os.makedirs(target_directory, exist_ok=True)
        target_cameras_file = os.path.join(target_directory, "cameras.bin")
        target_images_file = os.path.join(target_directory, "images.bin")
        shutil.copy(cameras_intrinsic_file, target_cameras_file)
        shutil.copy(cameras_extrinsic_file, target_images_file)
        
        pts_4_3dgs = np.concatenate([p[m.ravel()] for p, m in zip(pts3d, msk)])
        color_4_3dgs = np.concatenate([p[m] for p, m in zip(imgs, msk)])
        valid_msk = np.isfinite(pts_4_3dgs.sum(axis=1))
        color_4_3dgs = (color_4_3dgs * 255.0).astype(np.uint8)
        pts_4_3dgs = pts_4_3dgs[valid_msk]
        color_4_3dgs = color_4_3dgs[valid_msk]
        pts_4_3dgs = transform_pcl(pts_4_3dgs, R_e2o, t_e2o)
        storePly(os.path.join(filepath, f"{n_views}_views", "mast3r", "points3D.ply"), pts_4_3dgs, color_4_3dgs)
        
    else:
        # save TODO 并不完整
        save_colmap_cameras(ori_size, cam_intrinsics, os.path.join(output_colmap_path, 'cameras.txt'))
        save_colmap_images(camera_poses, os.path.join(output_colmap_path, 'images.txt'), train_img_list)
        pass
    
    return point_cloud, cam_intrinsics, camera_poses

def save_point_cloud(point_cloud, output_file):
    """Save point cloud as PLY file."""
    # pcd = trimesh.PointCloud(point_cloud)
    point_cloud.export(output_file)

# Example usage
if __name__ == "__main__":
    parser = get_args_parser()
    parser.add_argument("--n_views", type=int, default=3, help="Number of views")
    parser.add_argument("--output_colmap_path", type=str, default=None, help="Path to output COLMAP")
    parser.add_argument("--dataset_path", type=str, default=None, help="Path to imgs")
    parser.add_argument("--dataset_name", type=str, default=None, help="Name of dataset")
    parser.add_argument("--scene_name", type=str, default=None, help="Name of scene")
    parser.add_argument("--cache_dir", type=str, default=None, help="Path to cache directory")
    parser.add_argument("--llffhold", type=int, default=8, help="LLFF hold")
    parser.add_argument("--shared_intrinsics", action="store_true", help="Use shared intrinsics")
    parser.add_argument("--know_camera", action="store_true", help="Use known camera poses")
    args = parser.parse_args()

    filepath = os.path.join(args.dataset_path, args.dataset_name, args.scene_name)
    output_mast3r_path = os.path.join(filepath, "mast3r")
    image_size = 512
    if args.weights is not None:
        weights_path = args.weights
    else:
        weights_path = "naver/" + args.model_name
    model = AsymmetricMASt3R.from_pretrained(weights_path).to(args.device)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    optim_params = {
        "matching_conf_thr": 5.0
    }
    
    point_cloud, camera_intrinsics, camera_poses = reconstruct_scene(
        filepath, image_size, model, device, optim_params, args.cache_dir,
        args.output_colmap_path, args.n_views, args.shared_intrinsics,
        args.know_camera, min_conf_thr=2
    )
    
    
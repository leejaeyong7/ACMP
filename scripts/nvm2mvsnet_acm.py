import sys
sys.path.append('.')
sys.path.append('..')

import collections
from utils.colmap import qvec2rotmat
import struct
import numpy as np
import multiprocessing as mp
from functools import partial
import os
import argparse
import shutil
import cv2
from tqdm import tqdm
from PIL import Image
from pathlib import Path

def parse_cam_line(cam_line):
    '''
    Output format:
    <File name> <focal length> <quaternion WXYZ> <camera center> <radial distortion> 0
    '''
    tokens = cam_line.split(' ')
    assert len(tokens) == 11

    image_name = Path(tokens[0]).name
    f = float(tokens[1])
    q = np.array([float(v) for v in tokens[2:6]])
    c = np.array([float(v) for v in tokens[6:9]])
    rd = float(tokens[9])

    # create intrinsic matrix

    K = np.eye(3)
    K[0, 0] = f
    K[1, 1] = f

    R = qvec2rotmat(q)
    t = -R.t @ c

    E = np.eye(4)
    E[:3, :3] = R
    E[:3, 3:] = t

    return {
        'file': image_name,
        'K': K,
        'E': E,
        'R': R,
        't': t,
        'point_ids': [],
        'tokens': None,
        'src_ids': []
    }

def parse_point_line(point_line):
    '''
    <XYZ> <RGB> <number of measurements> [<Image index> <Feature Index> <xy>]
    '''
    tokens = point_line.split(' ')
    wp = np.array([float(v) for v in tokens[0:3]])
    num_m = float(tokens[6])
    measurements = []
    for mid in range(num_m):
        iid = tokens[mid * 4 + 7]
        fid = tokens[mid * 4 + 7]
        xy = tokens[mid * 4 + 7]
        measurements.append(iid)

    return wp, measurements

def calc_score(cameras, points, ref_i, src_i):
    ref_pids = cameras[ref_i]['point_ids']
    src_pids = cameras[src_i]['point_ids']
    shared_ids = [pid for pid in ref_pids if pid in src_pids]

    score = 0
    angles = []
    ref_E = cameras[ref_i]['E']
    src_E = cameras[src_i]['E']
    ref_R = ref_E[:3, :3]
    ref_t = ref_E[:3, 3:]
    src_R = src_E[:3, :3]
    src_t = src_E[:3, 3:]

    ref_C = -ref_R.T @ ref_t
    src_C = -src_R.T @ src_t

    for pid in shared_ids:
        p = points[pid]
        norm_ref_d = (ref_C - p) / np.linalg.norm(ref_C - p)
        norm_src_d = (src_C - p) / np.linalg.norm(src_C - p)
        theta = (180 / np.pi) * np.arccos(norm_ref_d @ norm_src_d)
        angles.append(theta)
        score += 1

    if len(angles) > 0:
        angles_sorted = sorted(angles)
        triangulationangle = angles_sorted[int(len(angles_sorted) * 0.75)]
        if triangulationangle < 1:
            score = 0.0
    return score

def compute_depth_ranges(camera, points, num_d=256, scale_d=1):
    pids = camera['point_ids']
    w_points = np.array([points[pid] for pid in pids])
    c_points = camera['R'] @ w_points + camera['t']
    zs = c_points[..., 2]
    min_d = np.quantile(zs, q=0.01) * 0.75
    max_d = np.quantile(zs, q=0.99) * 1.25
    int_d = (max_d - min_d) / (num_d - 1) / scale_d 
    tokens = (min_d, int_d, num_d, max_d)
    camera['tokens'] = tokens


def parse_nvm(nvm_file):
    '''
    NVM_V3 <- line 0

    <Number of cameras>  <- line 2
    <List of cameras> <- line 2 + num_cam

    <Number of 3D points> <- line 4 + num_cam
    <List of points> <- line 4 + num_cam + num_points
    0 <- line 5 + num_cam + num_points

    # output
    cameras: {
        cid: {
            K, E, point_ids
        }
    }
    points: {
        pid: xyz
    }
    '''
    with open(nvm_file, 'r') as f:
        lines = f.readlines()
    assert lines[0] == 'NVM_V3'

    # iterate for cameras
    cameras = {}
    points = {}
    num_camera = int(lines[2])
    for cid in range(num_camera):
        cam_line = lines[2 + cid]
        camera = parse_cam_line(cam_line)
        cameras[cid] = camera

    # iterate for points
    num_points = int(lines[4 + num_camera])
    for pid in range(num_points):
        point_line = lines[4 + pid]
        point, measurements = parse_point_line(point_line)
        points[pid] = point
        for m in measurements:
            cameras[m]['point_ids'].append(m)

    # compute depth ranges
    for camera in cameras.values():
        compute_depth_ranges(camera, points)

    # compute scores
    cids = list(cameras.keys())
    score_maps = np.zeros(len(cids), len(cids))
    for ref_cid in range(cids):
        for src_cid in range(ref_cid, cids):
            score = calc_score(cameras, points, ref_cid, src_cid)
            score_maps[ref_cid, src_cid] = score
            score_maps[src_cid, ref_cid] = score
    num_view = min(20, len(cids) - 1)
    for cid in cids:
        sorted_score = np.argsort(score_maps[cid])[::-1]
        cameras[cid]['src_ids'] = [(src_id, score_maps[cid, src_id]) for src_id in sorted_score[:num_view]]

    return cameras, points

def write_pair_file(pair_path, cameras):
    with open(pair_path, 'w') as f:
        f.write('%d\n' % len(cameras.keys()))
        for cid, camera in cameras.items():
            f.write('%d\n%d ' % (cid, len(camera['src_ids'])))
            for (src_id, src_score) in camera['src_ids']:
                f.write('%d %d ' % (src_id, src_score))
            f.write('\n')
    return

def write_cam_file(cam_path, camera):
    E = camera['E']
    K = camera['K']
    tokens = camera['tokens']
    with open(cam_path, 'w') as f:
        f.write('extrinsic\n')
        for j in range(4):
            for k in range(4):
                f.write(str(E[j, k]) + ' ')
            f.write('\n')
        f.write('\nintrinsic\n')
        for j in range(3):
            for k in range(3):
                f.write(str(K[j, k]) + ' ')
            f.write('\n')
        f.write('\n%f %f %f %f\n' % (tokens[0], tokens[1], tokens[2], tokens[3]))
    return

def write_img_file(img_path, input_img_dir, camera):
    input_img_path = camera['file'] / input_img_dir
    if(input_img_path.name.endswith('.jpg')):
        shutil.copyfile(input_img_path, img_path)
    else:
        Image.open(input_img_path).save(img_path)



def main(args):
    # setup paths
    scene_path = Path(args.scene_folder)
    nvm_path = scene_path / 'reconstruction0.nvm'
    input_img_dir = scene_path / 'images'

    output_path = Path(args.output_folder)
    output_path.mkdir(exist_ok=True, parents=True)

    # read nvm file
    cameras, points = parse_nvm(nvm_path)

    cam_dir = output_path / 'cams'
    img_dir = output_path / 'images'
    pair_path = output_path / 'pair.txt'

    cam_dir.mkdir(exist_ok=True)
    img_dir.mkdir(exist_ok=True)

    # write pair file
    write_pair_file(pair_path, cameras)
    for cid, camera in cameras.items():
        cam_path  = cam_dir / f'{cid:08d}_cam.txt'
        img_path = img_dir / f'{cid:08d}.jpg'

        write_cam_file(cam_path, camera)
        write_img_file(img_path, input_img_dir, camera)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert colmap camera')

    parser.add_argument('--scene_folder', required=True, type=str, help='Input Scene Folder.')
    parser.add_argument('--output_folder', required=True, type=str, help='save_folder.')

    parser.add_argument('--max_d', type=int, default=192)
    parser.add_argument('--interval_scale', type=float, default=1)

    args = parser.parse_args()
    main(args)
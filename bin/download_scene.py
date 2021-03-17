import sys
sys.path.append('.')
sys.path.append('..')

# local imports
from utils.clients.s3_client import S3Client
import json
from argparse import ArgumentParser
from tqdm import tqdm
from pathlib import Path
import random
random.seed(0)


def main(args):
    scene_id = args.scene_id
    project_id = args.project_id
    output_path = Path(args.output_path)

    with open(args.environment_path) as f:
        settings_dict = json.load(f)
    s3_client = S3Client(settings_dict)
    pcids = s3_client.list_pointclouds(project_id)
    if(scene_id not in pcids):
        raise RuntimeError(f"No Point Cloud {scene_id} Found!")

    # fetch images
    out_scene_path = output_path / scene_id
    s3_client.download_images(out_scene_path, project_id, scene_id)
    s3_client.download_nvm(out_scene_path, project_id, scene_id)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--output_path', type=str, default='./data', help='Path to download data to')
    parser.add_argument('--environment_path', type=str, default='./configs/environment.json', help='AWS Related Settings')
    parser.add_argument('--project_id', type=str, required=True, help='ObjectID for the Project')
    parser.add_argument('--scene_id', type=str, required=True, help='Object ID for the point cloud')
    args = parser.parse_args()
    main(args)
import boto3
import json
from botocore.config import Config

class S3Client:
    def __init__(self, settings):
        self.settings = settings
        self.client = boto3.client('s3',
            aws_access_key_id=settings['id'],
            aws_secret_access_key=settings['password']
        )
    
    def list_pointclouds(self, project_id):
        res = self.client.list_objects_v2(
            Bucket=self.settings['bucket'],
            Prefix="{}/PointCloud/".format(project_id),
            Delimiter = "/"
        )
        return [p['Prefix'].split('/')[-2] for p in res['CommonPrefixes']]

    def get_images(self, project_id, pointcloud_id):
        try:
            res = self.client.get_object(
                Bucket=self.settings['bucket'],
                Key="{}/PointCloud/{}/cameras.out".format(project_id, pointcloud_id)
            )
        except:
            print('PointCloud {} is missing cameras.out'.format(pointcloud_id))
            return []
        stream = res['Body']
        cameras_json = json.loads(stream.read().decode("utf-8"))
        images = []
        for camera in cameras_json:
            cid = list(camera.keys())[0]
            cam = camera[cid]
            image = cam['Image']
            images.append(image)
        return images

    def download_nvm(self, local_path, project_id, pointcloud_id):
        s3_pc_root = f'{project_id}/PointCloud/{pointcloud_id}'
        nvm_file = 'reconstruction0.nvm'
        nvm_path = local_path /  nvm_file
        self.client.download_file(self.settings['bucket'], s3_pc_root + '/' + nvm_file, str(nvm_path))

    def download_images(self, local_path, project_id, pointcloud_id):
        images = self.get_images(project_id, pointcloud_id)
        image_path = local_path /  'images'
        image_path.mkdir(exist_ok=True, parents=True)
        for image in images:
            s3_path = f'{project_id}/PointCloud/{pointcloud_id}/undistorted/{image}'
            self.client.download_file(self.settings['bucket'], s3_path, str(image_path/ image))

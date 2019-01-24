import re
import os

from rastervision.utils.files import list_paths

# You may need to adjust these URIs.
remote_root_uri = 's3://raster-vision-lf-dev/noisy-buildings-semseg/'
local_root_uri = '/opt/data/noisy-buildings-semseg/'
remote_raw_data_uri = ('s3://spacenet-dataset/SpaceNet_Buildings_Dataset_Round2/'
                       'spacenetV2_Train/AOI_2_Vegas')
local_raw_data_uri = '/opt/data/AOI_2_Vegas_Train'


def get_root_uri(use_remote_data):
    return remote_root_uri if use_remote_data else local_root_uri


def get_raw_data_uri(use_remote_data):
    return remote_raw_data_uri if use_remote_data else local_raw_data_uri


def get_exp_id(noise_mode, run):
    return '{}-{}'.format(noise_mode, run)


class NoiseMode():
    DROP = 'drop'
    SHIFT = 'shift'

    def __init__(self, type, level):
        self.type = type
        self.level = level

    def __repr__(self):
        return '{}-{}'.format(self.type, self.level)


class VegasBuildings():
    def __init__(self, use_remote_data):
        self.root_uri = get_root_uri(use_remote_data)
        self.raw_data_uri = get_raw_data_uri(use_remote_data)

        self.raster_dir = 'RGB-PanSharpen'
        self.label_dir = 'geojson/buildings'
        self.raster_fn_prefix = 'RGB-PanSharpen_AOI_2_Vegas_img'
        self.label_fn_prefix = 'buildings_AOI_2_Vegas_img'

    def get_class_map(self):
        return {
            'Building': (1, 'orange'),
            'Background': (2, 'black')
        }

    def get_class_id_to_filter(self):
        return {1: ['has', 'building']}

    def get_raster_source_uri(self, id):
        return os.path.join(
            self.raw_data_uri, self.raster_dir,
            '{}{}.tif'.format(self.raster_fn_prefix, id))

    def get_geojson_uri(self, id):
        return os.path.join(
            self.raw_data_uri, self.label_dir,
            '{}{}.geojson'.format(self.label_fn_prefix, id))

    def get_noisy_geojson_uri(self, noise_mode, id):
        return os.path.join(
            self.root_uri, 'noisy-labels', str(noise_mode),
            '{}{}.geojson'.format(self.label_fn_prefix, id))

    def get_scene_ids(self):
        label_dir = os.path.join(self.raw_data_uri, self.label_dir)
        label_paths = list_paths(label_dir, ext='.geojson')
        label_re = re.compile(r'.*{}(\d+)\.geojson'.format(
            self.label_fn_prefix))
        scene_ids = [
            label_re.match(label_path).group(1)
            for label_path in label_paths]
        return scene_ids

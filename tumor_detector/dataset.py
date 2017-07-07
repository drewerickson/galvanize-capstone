import config
import os
import boto
import nibabel as nib
import numpy as np


def connect_to_bucket(bucket_name):
    """
    Connects to the S3 bucket indicated with bucket_name.
    Returns a boto connection.
    """

    conn = boto.connect_s3(config.access_key, config.secret_access_key)
    return conn.get_bucket(bucket_name)


def load_nifti_file(key, local=True, all_dims=True):
    """
    Loads a NIfTI file at the given key.  Allows local or remote loading.  Reduces 3rd dimension if all_dims is False.
    Returns a 2D or 3D numpy array.
    """

    if local:
        data = load_nifti_file_local(key)
    else:
        data = load_nifti_file_S3(key)
    if not all_dims:
        data = data[:, :, 80]
    return data


def load_nifti_file_S3(key):
    """
    Loads a NIfTI file from S3 at the given key.
    Returns a 3D numpy array.
    """

    with open("temp.nii.gz", "wb") as f:
        f.write(key.read())
        f.close()
    data = load_nifti_file_local("temp.nii.gz")
    os.remove("temp.nii.gz")
    return data


def load_nifti_file_local(file_path):
    """
    Loads a NIfTI file from local folder at the given key (full path).
    Returns a 3D numpy array.
    """

    image = nib.load(file_path)
    data = image.get_data()
    return data


def category_pass(data, multi_cat=True):
    """
    If multi_cat is True, runs a pass to reduce all nonzero values to 1 (for a single category).
    Takes in data, a numpy array.
    Returns a numpy array.
    """

    if not multi_cat:
        data[data > 0] = 1
    return data


class DataSet(object):
    """
    DataSet object holds X and y datasets and related variables.
    Assumptions about folder structure are made.

    bucket: S3 bucket
    prefix_folder: string of folder in S3 bucket where data is stored
    local_path: string of local file path where data is stored
    X: numpy array of X data
    y: numpy array of y data
    index_train: numpy array of booleans indicating which data to use for training.
    """

    def __init__(self, bucket_name=None, prefix_folder=None, local_path=None):

        self.bucket = connect_to_bucket(bucket_name) if bucket_name else None
        self.prefix_folder = prefix_folder
        self.local_path = local_path

        self.X = []
        self.y = []
        self.index_train = None

    def load_dataset(self, local=True, all_dims=True, multi_cat=True):
        """
        Loads X and y datasets with specified parameters.
        local: If True, load data from local_path. If False, load from S3.
        all_dims: If True, keep z dimension.  If False, select one z slice (defined in function).
        multi_cat: If True, keep all categories in y.  If False, set all categories to 1.
        """

        X_file_name = "t1.nii.gz"  # only one channel to start
        y_file_name = "seg.nii.gz"
        X_keys, y_keys = self.get_keys(X_file_name, y_file_name, local=local)

        self.X = np.stack(
            [np.expand_dims(load_nifti_file(key,
                                            local=local,
                                            all_dims=all_dims),
                            axis=-1) for key in X_keys],
            axis=0)
        self.y = np.stack(
            [np.expand_dims(category_pass(load_nifti_file(key,
                                                          local=local,
                                                          all_dims=all_dims),
                                          multi_cat=multi_cat),
                            axis=-1) for key in y_keys],
            axis=0)

    def get_keys(self, X_file_name, y_file_name, local=True):
        """
        Gets 'keys' based on X and y file names.
        local: If True, gets local file paths as 'keys'.  If False, get S3 bucket keys.
        """

        X_keys = []
        y_keys = []
        if local:
            for root, dirs, files in os.walk(self.local_path):
                for file_name in files:
                    if file_name == X_file_name:
                        X_keys.append(os.path.join(root, file_name))
                    elif file_name == y_file_name:
                        y_keys.append(os.path.join(root, file_name))
        else:
            for key in self.bucket.list(prefix=self.prefix_folder):
                if X_file_name in key.name:
                    X_keys.append(key)
                elif y_file_name in key.name:
                    y_keys.append(key)

        return X_keys[0:3], y_keys[0:3]  # truncated for testing

    def test_train_split(self):
        """
        Future method for selecting a subset of X and y for a test / train split.
        """
        pass


if __name__ == '__main__':
    ds = DataSet(local_path=config.local_path)
    ds.load_dataset()
    #    ds.load_dataset(all_dims=False)
    #    ds.load_dataset(multi_cat=False)
    #    ds.load_dataset(all_dims=False, multi_cat=False)
    #    ds = DataSet(config.bucket_name, "train")
    #    ds.load_dataset(local=False)
    #    ds.load_dataset(local=False, all_dims=False)
    #    ds.load_dataset(local=False, multi_cat=False)
    #    ds.load_dataset(local=False, all_dims=False, multi_cat=False)

    print("Complete")

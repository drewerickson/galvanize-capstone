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
    return image.get_data()


def save_nifti_file_S3(data, file_path, bucket):
    """
    Saves a NIfTI file to S3 at the given key.
    """

    key = bucket.new_key(file_path)
    save_nifti_file_local(data, "temp.nii.gz")
    with open("temp.nii.gz", "rb") as f:
        key.set_contents_from_file(f)
    os.remove("temp.nii.gz")


def save_nifti_file_local(data, file_path):
    """
    Saves a NIfTI file to a local folder at the given key (full path).
    """

    image = nib.Nifti1Image(data, affine=np.eye(4))
    nib.save(image, file_path)


def contrast_pass(data, multi_cat=True):
    """
    Combines contrasts together in a single numpy array.
    If only one data element, returns an array with an expanded dimension.
    Takes in data_list, a list of numpy arrays.
    Returns a numpy array.
    """
    new_data = np.array(data)
    #
    return new_data


def category_pass(data, multi_cat=True):
    """
    Splits label into multiple categories.
    If multi_cat is True, runs a pass to reduce all nonzero values to 1 (for a single category).
    Takes in data, a numpy array.
    Returns a numpy array.
    """

    data_list = []
    if not multi_cat:
        data_list.append((data != 0)*1)
    else:
        data_list.extend([(data == 1)*1,
                          (data == 2)*1,
                          (data == 3)*1,
                          (data == 4)*1])
    return np.stack(data_list, axis=-1)


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

        self.X_file_names = ["t1.nii.gz", "t2.nii.gz", "t1ce.nii.gz", "flair.nii.gz"]
        self.y_file_name = "seg.nii.gz"
        self.y_predict_file_name = "predict.nii.gz"

        if bucket_name and prefix_folder:
            self.bucket = connect_to_bucket(bucket_name) if bucket_name else None
            self.prefix_folder = prefix_folder
            self.local = False
        elif local_path:
            self.local_path = local_path
            self.local = True

        self.X_keys = None
        self.y_keys = None
        self.X = []
        self.y = []
        self.index_train = None
        self.index_test = None

        self.y_predict = []

    def load_dataset(self, all_dims=True, multi_cat=True):
        """
        Loads X and y datasets with specified parameters.
        local: If True, load data from local_path. If False, load from S3.
        all_dims: If True, keep z dimension.  If False, select one z slice (defined in function).
        multi_cat: If True, keep all categories in y.  If False, set all categories to 1.
        """

        self.get_keys()

        self.get_X(all_dims)
        self.get_y(all_dims, multi_cat)
        self.mask_nz()

        self.train_test_split()

    def get_keys(self):
        """
        Gets 'keys' based on X and y file names.
        local: If True, gets local file paths as 'keys'.  If False, get S3 bucket keys.
        """

        X_keys = [[] for _ in self.X_file_names]
        y_keys = []
        if self.local:
            for root, dirs, files in os.walk(self.local_path):
                for file_name in files:
                    if file_name in self.X_file_names:
                        X_keys[self.X_file_names.index(file_name)].append(os.path.join(root, file_name))
                    elif file_name == self.y_file_name:
                        y_keys.append(os.path.join(root, file_name))
        else:
            for key in self.bucket.list(prefix=self.prefix_folder):
                file_name = key.name.split("/")[-1]
                if file_name in self.X_file_names:
                    X_keys[self.X_file_names.index(file_name)].append(key)
                elif self.y_file_name in key.name:
                    y_keys.append(key)

        self.X_keys = [subkeys[:20] for subkeys in X_keys]
        self.y_keys = y_keys[:20]  # truncated for testing

    def get_X(self, all_dims):
        X = []
        for index_row in range(len(self.X_keys[0])):
            X_row = []
            for index_contrast in range(len(self.X_keys)):
                X_row.append(self.load_nifti_file(self.X_keys[index_contrast][index_row], all_dims=all_dims))
            X.append(np.stack(X_row, axis=-1))
        self.X = np.stack(X, axis=0)

    def get_y(self, all_dims, multi_cat):
        self.y = np.stack([category_pass(self.load_nifti_file(key,
                                                              all_dims=all_dims),
                                         multi_cat=multi_cat) for key in self.y_keys],
                          axis=0)

    def mask_nz(self):
        """
        For each X, create an exclusive mask for all voxels that have all nonzero X values.
        Set the mask to zero for all y values, and add the mask as another y label.
        The result will be a label for all nonzero, non categorized values.
        """
        y_update = []
        for i, X_entry in enumerate(self.X):
            y_nz = (X_entry[..., 0] != 1) * 1
            for index_scan in range(X_entry.shape[-1]):
                y_nz *= ((X_entry[..., index_scan] != 0) * 1)
            for index_cat in range(self.y[i].shape[-1]):
                y_nz *= ((self.y[i][..., index_cat] == 0) * 1)
            y_nz = np.expand_dims(y_nz, axis=-1)
            y_update.append(np.append(self.y[i], y_nz, axis=-1))
        self.y = np.stack(y_update, axis=0)

    def load_nifti_file(self, key, all_dims=True):
        """
        Loads a NIfTI file at the given key.  Allows local or remote loading.
        Reduces 3rd dimension if all_dims is False.
        Returns a 2D or 3D numpy array.
        """

        if self.local:
            data = load_nifti_file_local(key)
        else:
            data = load_nifti_file_S3(key)
        if not all_dims:
            data = data[40:200, 41:217, 80]  # crop close to brain, selected z plane 80 for testing
        else:
            data = data[40:200, 41:217, 1:145]  # crop with some zero buffer
        return data

    def save_nifti_file(self, data, file_path):
        """
        Save NIfTI data to file (local or S3).
        """

        if self.local:
            save_nifti_file_local(data, file_path)
        else:
            save_nifti_file_S3(data, file_path, self.bucket)

    def save_y_predict(self, model):
        """
        Save the y prediction data to files (local or S3).
        """
        if len(self.y_predict) == 0:
            self.y_predict = self.run_predict(model)
        for i, y_key in enumerate(self.y_keys):
            if self.local:
                y_predict_file_path = "/".join(y_key.split("/")[:-1]) + "/" + self.y_predict_file_name
            else:
                y_predict_file_path = "/".join(y_key.name.split("/")[:-1]) + "/" + self.y_predict_file_name
            self.save_nifti_file(self.y_predict[i], y_predict_file_path)

    def run_predict(self, model):
        """
        Save the y prediction data to files (local or S3).
        """
        return model.predict(self.X)

    def train_test_split(self, train_pct=0.8):
        """
        Creates a random set of indices for selecting a subset of X and y for a train / test split.
        """
        self.index_train = np.random.choice(len(self.X), int(np.round(len(self.X)*train_pct)), replace=False)
        self.index_test = np.array([i for i in np.arange(len(self.X)) if i not in self.index_train])

    def X_train(self):
        return self.X[self.index_train]

    def X_test(self):
        return self.X[self.index_test]

    def y_train(self):
        return self.y[self.index_train]

    def y_test(self):
        return self.y[self.index_test]


if __name__ == '__main__':
    ds = DataSet(local_path=config.local_path)
#    ds.load_dataset()
#    ds.load_dataset(all_dims=False)
#    ds.load_dataset(multi_cat=False)
    ds.load_dataset(all_dims=False)
    for i in range(len(ds.X)):
        save_path = "/".join(ds.y_keys[i].split("/")[:-1])
        ds.save_nifti_file(ds.X[i], save_path + "/X.nii.gz")
        ds.save_nifti_file(ds.y[i], save_path + "/y.nii.gz")
#    ds = DataSet(config.bucket_name, "train")
#    ds.load_dataset()
    #    ds.load_dataset(local=False, all_dims=False)
    #    ds.load_dataset(local=False, multi_cat=False)
#    ds.load_dataset(all_dims=False, multi_cat=False)
#    ds.save_nifti_file(ds.y[0], "/".join(ds.y_keys[0].name.split("/")[:-1]) + "/predict.nii.gz")

    print("Complete")

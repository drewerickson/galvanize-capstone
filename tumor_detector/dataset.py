import config
import os
import boto
import nibabel as nib
import numpy as np

# This is a curated set of patients that have greater than 1% voxels in each of the three tumor categories
robust_pids = ['Brats17_2013_11_1',
               'Brats17_2013_12_1',
               'Brats17_2013_17_1',
               'Brats17_2013_19_1',
               'Brats17_2013_21_1',
               'Brats17_2013_22_1',
               'Brats17_CBICA_AAP_1',
               'Brats17_CBICA_AQU_1',
               'Brats17_CBICA_AQV_1',
               'Brats17_CBICA_ARF_1',
               'Brats17_CBICA_ASA_1',
               'Brats17_CBICA_ASG_1',
               'Brats17_CBICA_ASV_1',
               'Brats17_CBICA_ATF_1',
               'Brats17_CBICA_AUN_1',
               'Brats17_CBICA_AWG_1',
               'Brats17_CBICA_AXM_1',
               'Brats17_CBICA_AXN_1',
               'Brats17_CBICA_AXO_1',
               'Brats17_TCIA_105_1',
               'Brats17_TCIA_118_1',
               'Brats17_TCIA_151_1',
               'Brats17_TCIA_167_1',
               'Brats17_TCIA_180_1',
               'Brats17_TCIA_184_1',
               'Brats17_TCIA_203_1',
               'Brats17_TCIA_222_1',
               'Brats17_TCIA_241_1',
               'Brats17_TCIA_242_1',
               'Brats17_TCIA_257_1',
               'Brats17_TCIA_265_1',
               'Brats17_TCIA_274_1',
               'Brats17_TCIA_296_1',
               'Brats17_TCIA_300_1',
               'Brats17_TCIA_335_1',
               'Brats17_TCIA_374_1',
               'Brats17_TCIA_390_1',
               'Brats17_TCIA_401_1',
               'Brats17_TCIA_410_1',
               'Brats17_TCIA_412_1',
               'Brats17_TCIA_419_1',
               'Brats17_TCIA_429_1',
               'Brats17_TCIA_430_1',
               'Brats17_TCIA_436_1',
               'Brats17_TCIA_444_1',
               'Brats17_TCIA_460_1',
               'Brats17_TCIA_469_1',
               'Brats17_TCIA_478_1',
               'Brats17_TCIA_603_1',
               'Brats17_TCIA_654_1']


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
        data_list.append((data != 0) * 1)
    else:
        data_list.extend([(data == 1) * 1,
                          (data == 2) * 1,
                          (data == 4) * 1])
    return np.stack(data_list, axis=-1)


class DataProcessor(object):
    """
    DataProcessor takes in data and runs a series of processes on it.
    For a given directory of files, it runs through a process for each patient.
    The results are saved to file in the same location as the original data.

    bucket: S3 bucket
    prefix_folder: string of folder in S3 bucket where data is stored
    local_path: string of local file path where data is stored
    X: numpy array of X data
    y: numpy array of y data
    index_train: numpy array of booleans indicating which data to use for training.
    """

    def __init__(self, bucket_name=None, prefix_folder=None, local_path=None):

        self.X_input_file_names = ["t1.nii.gz", "t2.nii.gz", "t1ce.nii.gz", "flair.nii.gz"]
        self.y_input_file_name = "seg.nii.gz"

        self.X_3d_file_name = "X-3D.nii.gz"
        self.y_3d_file_name = "y-3D.nii.gz"
        self.X_2d_x_file_name = "X-2D-x.nii.gz"
        self.X_2d_y_file_name = "X-2D-y.nii.gz"
        self.X_2d_z_file_name = "X-2D-z.nii.gz"
        self.y_2d_x_file_name = "y-2D-x.nii.gz"
        self.y_2d_y_file_name = "y-2D-y.nii.gz"
        self.y_2d_z_file_name = "y-2D-z.nii.gz"

        if bucket_name and prefix_folder:
            self.bucket = connect_to_bucket(bucket_name) if bucket_name else None
            self.prefix_folder = prefix_folder
            self.local = False
        elif local_path:
            self.local_path = local_path
            self.local = True

        self.X_keys = None
        self.y_keys = None
        self.X_3d = None
        self.y_3d = None
        self.X_2d_x = None
        self.X_2d_y = None
        self.X_2d_z = None
        self.y_2d_x = None
        self.y_2d_y = None
        self.y_2d_z = None

    def process_data(self):
        """
        Gets all the relevant patient 'keys'.
        For each patient 'key', runs the specified processors and saves the results.
        local: If True, load data from local_path. If False, load from S3.
        """

        print("Starting Process")
        print("Get Keys: ... ", end="", flush=True)
        self.get_keys()
        print("Done.")

        for i in range(140, len(self.X_keys)):

            self.X_3d = None
            self.y_3d = None
            self.X_2d_x = None
            self.X_2d_y = None
            self.X_2d_z = None
            self.y_2d_x = None
            self.y_2d_y = None
            self.y_2d_z = None

            print("Processing: ", str(i+1), " of ", str(len(self.X_keys)), " ... ", end="", flush=True)
            self.process_3d(i)
            self.process_2d()
            self.save_output(i)
            print("Done.")

        print("Processing Complete.")

    def get_keys(self):
        """
        Gets 'keys' based on X and y input file names.
        local: If True, gets local file paths as 'keys'.  If False, get S3 bucket keys.
        """

        X_keys = []
        y_keys = []
        if self.local:
            for root, dirs, files in os.walk(self.local_path):
                X_keygroup = [None] * len(self.X_input_file_names)
                for file_name in files:
                    if file_name in self.X_input_file_names:
                        X_keygroup[self.X_input_file_names.index(file_name)] = os.path.join(root, file_name)
                    elif file_name == self.y_input_file_name:
                        y_keys.append(os.path.join(root, file_name))
                if X_keygroup[0]:
                    X_keys.append(X_keygroup)
        else:
            for key in self.bucket.list(prefix=self.prefix_folder):
                file_name = key.name.split("/")[-1]
                patientID = key.name.split("/")[-2]
                if file_name in self.X_input_file_names:
                    X_keys[self.X_input_file_names.index(file_name)].append(key)
                elif self.y_input_file_name in key.name:
                    y_keys.append(key)

        # separate assignment step to allow subsampling
        self.X_keys = X_keys
        self.y_keys = y_keys

    def process_3d(self, i):

        # load the X input across all scans, crop/buffer the dims, and stack it
        X_keygroup = self.X_keys[i]
        X_3d = []
        for index_contrast in range(len(X_keygroup)):
            X_input = self.load_nifti_file(X_keygroup[index_contrast])
            X_3d.append(np.pad(X_input, ((0, 0), (0, 0),(45, 40)), 'constant', constant_values=0))  # z padding to 240
            # [40:200, 41:217, 1:145])  # crop with some zero buffer

        self.X_3d = np.stack(X_3d, axis=-1)

        # load the y input, crop/buffer the dims, split it by category, and stack it
        y_input = self.load_nifti_file(self.y_keys[i])
        y_3d = np.pad(y_input, ((0, 0), (0, 0), (45, 40)), 'constant', constant_values=0)
        y_3d = [(y_3d == 1) * 1, (y_3d == 2) * 1, (y_3d == 4) * 1]

        # generate a brain mask from X and y data, then add it
        y_nz = (X_3d[0] != 1) * 1
        for index_scan in range(len(X_3d)):
            y_nz *= ((X_3d[index_scan] != 0) * 1)
        for index_cat in range(len(y_3d)):
            y_nz *= ((y_3d[index_cat] == 0) * 1)
        y_3d.append(y_nz)

        self.y_3d = np.stack(y_3d, axis=-1)

    def process_2d(self):
        # for each axis, for each slice, calculate the total count for each category (including brain)
                # for each category (excluding brain), select the slice with the maximum total count
                # from those slices, select the one with the minimal difference between all categories
        # shape = self.y_3d[i].shape
        # y_2d_x_values = np.zeros(shape[0])
        # index_max = np.zeros(shape[-1])
        # for i_x in range(shape[0]):
        #     y_2d_x_value_group = np.zeros(shape[-1])
        #     for i_c in range(shape[-1]-1):
        #         contrast_count = np.count_nonzero((self.y[i][i_x, ..., i_c] == 1) * 1)
        #         y_2d_x_value_group[i_c] = contrast_count
        #         if contrast_count > index_max[i_c][1]:
        #             index_max[i_c] = (i_x, contrast_count)
        #     y_2d_x_values[i_x] = y_2d_x_value_group
        # index_best_slice = (0,self.y[i].size)
        # for i_max in range(index_max.shape[0]):
        #     value_group = y_2d_x_values[index_max[i_max][0]]
        #     diff = np.sum(np.abs(np.diff(value_group)))
        #     if diff < index_best_slice[1]:
        #         index_best_slice = (index_max[i_max][0], diff)
        i_x = self.get_optimal_slice_index(self.y_3d)
        self.X_2d_x = self.X_3d[i_x, :, :, :]
        self.y_2d_x = self.y_3d[i_x, :, :, :]
        i_y = self.get_optimal_slice_index(np.swapaxes(self.y_3d, 0, 1))
        self.X_2d_y = self.X_3d[:, i_y, :, :]
        self.y_2d_y = self.y_3d[:, i_y, :, :]
        i_z = self.get_optimal_slice_index(np.swapaxes(self.y_3d, 0, 2))
        self.X_2d_z = self.X_3d[:, :, i_z, :]
        self.y_2d_z = self.y_3d[:, :, i_z, :]

    def get_optimal_slice_index(self, volume):
        """
        Gets the optimal slice for the 0th dimension.  Assumes -1th dimension is categorical.
        Ignores the last category.
        Takes in a 3D volume with a categorical dimension.
        Returns an integer index for the 0th dimension.
        """
        shape = volume.shape
        shape_c = shape[-1]-1
        y_2d_values = [None] * shape[0]
        index_max = [(120, 0)] * shape_c  # default to the center axis if no optimal axis exists for a category
        for i_d in range(shape[0]):
            y_2d_value_group = np.zeros(shape_c)
            for i_c in range(shape_c):
                contrast_count = np.count_nonzero((volume[i_d, ..., i_c] == 1) * 1)
                y_2d_value_group[i_c] = contrast_count
                if contrast_count > index_max[i_c][1]:
                    index_max[i_c] = (i_d, contrast_count)
            y_2d_values[i_d] = y_2d_value_group
        index_best_slice = (120, 0)
        for i_max in range(shape_c):
            i_group, max_value = index_max[i_max]
            value_group = y_2d_values[i_group]
            diff = np.sum(value_group-max_value)
            if diff < index_best_slice[1]:
                index_best_slice = (index_max[i_max][0], diff)
        return index_best_slice[0]

    def save_output(self, i):

        patient_path = ""
        if self.local:
            patient_path = "/".join(self.y_keys[i].split("/")[:-1])
        else:
            patient_path = "/".join(self.y_keys[i].name.split("/")[:-1])

        self.save_nifti_file(self.X_3d, patient_path + "/" + self.X_3d_file_name)
        self.save_nifti_file(self.y_3d, patient_path + "/" + self.y_3d_file_name)
        self.save_nifti_file(self.X_2d_x, patient_path + "/" + self.X_2d_x_file_name)
        self.save_nifti_file(self.X_2d_y, patient_path + "/" + self.X_2d_y_file_name)
        self.save_nifti_file(self.X_2d_z, patient_path + "/" + self.X_2d_z_file_name)
        self.save_nifti_file(self.y_2d_x, patient_path + "/" + self.y_2d_x_file_name)
        self.save_nifti_file(self.y_2d_y, patient_path + "/" + self.y_2d_y_file_name)
        self.save_nifti_file(self.y_2d_z, patient_path + "/" + self.y_2d_z_file_name)

    def load_nifti_file(self, key):
        """
        Loads a NIfTI file at the given key.  Allows local or remote loading.
        Returns a 3D numpy array.
        """

        if self.local:
            data = load_nifti_file_local(key)
        else:
            data = load_nifti_file_S3(key)
        return data

    def save_nifti_file(self, data, file_path):
        """
        Save NIfTI data to file (local or S3).
        """

        if self.local:
            save_nifti_file_local(data, file_path)
        else:
            save_nifti_file_S3(data, file_path, self.bucket)


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
                if root.split("/")[-1] in robust_pids:
                    for file_name in files:
                        if file_name in self.X_file_names:
                            X_keys[self.X_file_names.index(file_name)].append(os.path.join(root, file_name))
                        elif file_name == self.y_file_name:
                            y_keys.append(os.path.join(root, file_name))
        else:
            for key in self.bucket.list(prefix=self.prefix_folder):
                file_name = key.name.split("/")[-1]
                patientID = key.name.split("/")[-2]
                if patientID in robust_pids:
                    if file_name in self.X_file_names:
                        X_keys[self.X_file_names.index(file_name)].append(key)
                    elif self.y_file_name in key.name:
                        y_keys.append(key)

        self.X_keys = [subkeys for subkeys in X_keys]
        self.y_keys = y_keys

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
        self.index_train = np.random.choice(len(self.X), int(np.round(len(self.X) * train_pct)), replace=False)
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
    dp = DataProcessor(local_path=config.local_path)
    dp.process_data()
    #    ds = DataSet(local_path=config.local_path)
    #    ds.load_dataset()
    #    ds.load_dataset(all_dims=False)
    #    ds.load_dataset(multi_cat=False)
#    ds.load_dataset(all_dims=False)
#    for i in range(len(ds.X)):
#        save_path = "/".join(ds.y_keys[i].split("/")[:-1])
#        ds.save_nifti_file(ds.X[i], save_path + "/X.nii.gz")
#        ds.save_nifti_file(ds.y[i], save_path + "/y.nii.gz")
#    ds = DataSet(config.bucket_name, "train")
#    ds.load_dataset()
    #    ds.load_dataset(local=False, all_dims=False)
    #    ds.load_dataset(local=False, multi_cat=False)
    #    ds.load_dataset(all_dims=False, multi_cat=False)
    #    ds.save_nifti_file(ds.y[0], "/".join(ds.y_keys[0].name.split("/")[:-1]) + "/predict.nii.gz")

    print("Complete")

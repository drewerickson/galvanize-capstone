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


def save_nifti_file_S3(data, key):
    """
    Saves a NIfTI file to S3 at the given key.
    """

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

        for i in range(0, len(self.X_keys)):

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
            min = value_group.min()
            if min > index_best_slice[1]:
                index_best_slice = (index_max[i_max][0], min)
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

    local: Boolean to indicate whether the data is local or in S3.
    bucket: S3 bucket object
    prefix_folder: string of folder in S3 bucket where data is stored
    local_path: string of local file path where data is stored

    data: dictionary structure with patient data.
    { patient : 
        { data type : 
            { data bin : 
                { key :
                  data : 
                }
            }
        }
    }

    index_train: numpy array of booleans indicating which data to use for training.
    """

    def __init__(self, bucket_name=None, prefix_folder=None, local_path=None):

        self.data_types = ["2D-x", "2D-y", "2D-z"]
        self.data_bins = ["X", "y"]
        self.predict_bin = "predict"
        self.file_type = ".nii.gz"

        if bucket_name and prefix_folder:
            self.local = False
            self.bucket = connect_to_bucket(bucket_name) if bucket_name else None
            self.prefix_folder = prefix_folder
        elif local_path:
            self.local = True
            self.local_path = local_path

        self.data = {}

        self.train_patients = None
        self.test_patients = None

    def load_dataset(self, patients):
        """
        Loads X and y datasets for the specified patient IDs.
        """

        for patient in patients:
            new_patient = {}
            for data_type in self.data_types:
                new_type = {}
                for data_bin in self.data_bins:
                    key = self.get_key(patient, data_type, data_bin)
                    data = self.load_nifti_file(key)
                    new_type.update({data_bin: {"key": key, "data": data}})
                new_patient.update({data_type: new_type})
            self.data.update({patient: new_patient})

    def get_key(self, patient, data_type, data_bin):
        full_path = self.get_full_path(patient, data_type, data_bin)
        if self.local:
            return full_path
        else:
            key = self.bucket.get_key(full_path)
            if not key:
                key = self.bucket.new_key(full_path)
            return key

    def get_full_path(self, patient, data_type, data_bin):
        if self.local:
            return self.local_path + patient + "/" + self.get_file_name(data_type, data_bin)
        else:
            return self.prefix_folder + patient + "/" + self.get_file_name(data_type, data_bin)

    def get_file_name(self, data_type, data_bin):
        return data_bin + "-" + data_type + self.file_type

    def load_nifti_file(self, key):
        """
        Loads a NIfTI file at the given key.  Allows local or remote loading.
        Reduces 3rd dimension if all_dims is False.
        Returns a 2D or 3D numpy array.
        """

        if self.local:
            data = load_nifti_file_local(key)
        else:
            data = load_nifti_file_S3(key)
        return data

    def X(self):
        return self.get_data_subset(self.data.keys(), "X")

    def y(self):
        return self.get_data_subset(self.data.keys(), "y")

    def train_test_split(self, train_pct=0.8):
        """
        Creates a random set of patient IDs for selecting a subset of patient data for a train / test split.
        """
        index_train = np.random.choice(len(self.data), int(np.round(len(self.data) * train_pct)), replace=False)
        index_test = [i for i in np.arange(len(self.data)) if i not in index_train]
        patients = list(self.data.keys())
        self.train_patients = [patients[i] for i in index_train]
        self.test_patients = [patients[i] for i in index_test]

        return self.X_train(), self.X_test(), self.y_train(), self.y_test()

    def X_train(self):
        return self.get_data_subset(self.train_patients, "X")

    def X_test(self):
        return self.get_data_subset(self.test_patients, "X")

    def y_train(self):
        return self.get_data_subset(self.train_patients, "y")

    def y_test(self):
        return self.get_data_subset(self.test_patients, "y")

    def get_data_subset(self, patients, data_bin):
        """
        Creates a numpy array of data from a subset of patients.
        """

        data_subset = []
        for patient in patients:
            for data_type in self.data_types:
                data_subset.append(self.data[patient][data_type][data_bin]["data"])
        return np.stack(data_subset, axis=0)

    def predict(self, model, model_id):
        """
        Save the y prediction data to files (local or S3).
        """

        predict_array = model.predict(self.X())
        predicts = []
        for i in range(predict_array.shape[0]):
            predicts.append(predict_array[i])

        for patient in self.data.keys():
            for data_type in self.data_types:
                key = self.get_key(patient, data_type, self.predict_bin)
                data = predicts.pop(0)
                new_bin = {self.predict_bin: {"key": key, "data": data}}
                self.data[patient][data_type].update(new_bin)
                self.save_nifti_file(data, key)

    def save_nifti_file(self, data, key):
        """
        Save NIfTI data to file (local or S3).
        """

        if self.local:
            save_nifti_file_local(data, key)
        else:
            save_nifti_file_S3(data, key)


if __name__ == '__main__':

    local = False
    process = False

    if local and process:
        dp = DataProcessor(local_path=config.local_path)
        dp.process_data()
    elif local and (not process):
        ds = DataSet(local_path=config.local_path)
        ds.load_dataset(config.pids_of_interest)
        X_train, X_test, y_train, y_test = ds.train_test_split()
        X = ds.X()
        y = ds.y()
    elif (not local) and process:
        dp = DataProcessor(config.bucket_name, config.prefix_folder)
        dp.process_data()
    elif not(local and process):
        ds = DataSet(config.bucket_name, config.prefix_folder)
        ds.load_dataset(config.pids_of_interest)
        X_train, X_test, y_train, y_test = ds.train_test_split()
        X = ds.X()
        y = ds.y()

    print("Complete.")

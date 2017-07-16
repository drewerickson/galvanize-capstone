import config
import os
import boto
import nibabel as nib
import numpy as np
from skimage import exposure


def connect_to_bucket(bucket_name):
    """
    Connects to the S3 bucket with the given bucket name.
    Returns a boto connection.
    """
    conn = boto.connect_s3(config.access_key, config.secret_access_key)
    return conn.get_bucket(bucket_name)


def load_nifti_file_S3(key):
    """
    Loads a NIfTI file from S3 with the given key.
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
    Loads a NIfTI file from a local folder at the given file path.
    Returns a 3D numpy array.
    """
    image = nib.load(file_path)
    return image.get_data()


def save_nifti_file_S3(data, key):
    """
    Saves a NIfTI file to S3 with the given key.
    """
    save_nifti_file_local(data, "temp.nii.gz")
    with open("temp.nii.gz", "rb") as f:
        key.set_contents_from_file(f)
    os.remove("temp.nii.gz")


def save_nifti_file_local(data, file_path):
    """
    Saves a NIfTI file to a local folder at the given file path.
    """
    image = nib.Nifti1Image(data, affine=np.eye(4))
    nib.save(image, file_path)


class DataProcessor(object):
    """
    DataProcessor takes in data and runs a series of processes on it.
    For a given directory of files, it runs through a process for each patient (a folder in the root directory).
    The results are saved to file in the same location as the original data.
    """

    def __init__(self, bucket_name=None, prefix_folder=None, local_path=None):
        """
        bucket_name: S3 bucket name
        prefix_folder: string of folder in S3 bucket where data is stored
        local_path: string of local file path where data is stored
        """
        self.X_input_file_names = ["t1.nii.gz", "t2.nii.gz", "t1ce.nii.gz", "flair.nii.gz"]
        self.y_input_file_name = "seg.nii.gz"

        self.X_3d_file_name = "X-3D.nii.gz"
        self.y_3d_file_name = "y-3D.nii.gz"
        self.X_2d_x_file_name = "X-2D-x-core.nii.gz"
        self.X_2d_y_file_name = "X-2D-y-core.nii.gz"
        self.X_2d_z_file_name = "X-2D-z-core.nii.gz"
        self.y_2d_x_file_name = "y-2D-x-core.nii.gz"
        self.y_2d_y_file_name = "y-2D-y-core.nii.gz"
        self.y_2d_z_file_name = "y-2D-z-core.nii.gz"

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
        """
        print("Starting Process")
        print("Get Keys... ", end="", flush=True)
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
                if file_name in self.X_input_file_names:
                    X_keys[self.X_input_file_names.index(file_name)].append(key)
                elif self.y_input_file_name in key.name:
                    y_keys.append(key)

        # separate assignment step to allow subsampling
        self.X_keys = X_keys
        self.y_keys = y_keys

    def process_3d(self, i):
        """
        Process 3D arrays of X and y data.
        Loads the data using the keys at the given index in the X and y list.
        Stores the results temporarily in class variables.
        """
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
        """
        Select optimal 2D arrays of X and y data from each dimension of the 3D arrays.
        Stores the results temporarily in class variables.
        """
        i_x = self.get_optimal_core_slice_index(self.y_3d)
        self.X_2d_x = self.X_3d[i_x, :, :, :]
        self.y_2d_x = self.y_3d[i_x, :, :, :]
        i_y = self.get_optimal_core_slice_index(np.swapaxes(self.y_3d, 0, 1))
        self.X_2d_y = self.X_3d[:, i_y, :, :]
        self.y_2d_y = self.y_3d[:, i_y, :, :]
        i_z = self.get_optimal_core_slice_index(np.swapaxes(self.y_3d, 0, 2))
        self.X_2d_z = self.X_3d[:, :, i_z, :]
        self.y_2d_z = self.y_3d[:, :, i_z, :]

    def get_optimal_slice_index(self, volume):
        """
        Gets the optimal slice for the 0th dimension.  Optimal here is the slice that first has the maximum voxel count
        for all categories, then has the maximum voxel count for the category with the smallest count.
        Assumes -1th dimension is categorical.
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

    def get_optimal_core_slice_index(self, volume):
        """
        Gets the optimal core slice for the 0th dimension.  Optimal here is the slice that has the maximum count for the
        0th category.
        Assumes -1th dimension is categorical.
        Ignores the last category.
        Takes in a 3D volume with a categorical dimension.
        Returns an integer index for the 0th dimension.
        """
        shape = volume.shape
        index_max = (120, 0)  # default to the center axis if no optimal axis exists for a category
        for i_d in range(shape[0]):
            contrast_count = np.count_nonzero((volume[i_d, ..., 0] == 1) * 1)
            if contrast_count > index_max[1]:
                index_max = (i_d, contrast_count)
        return index_max[0]

    def save_output(self, i):
        """
        Saves the data arrays for all the relevant data objects.
        """
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
        Loads a NIfTI file at the given 'key'.  Allows local or remote loading, based on class boolean.
        Returns a numpy array.
        """
        if self.local:
            data = load_nifti_file_local(key)
        else:
            data = load_nifti_file_S3(key)
        return data

    def save_nifti_file(self, data, key):
        """
        Save NIfTI data array to file with the 'key'.  Allows local or remote saving, based on class boolean.
        """
        if self.local:
            save_nifti_file_local(data, key)
        else:
            save_nifti_file_S3(data, key)


class DataSet(object):
    """
    DataSet object holds X and y data sets and related variables.
    Assumptions about folder structure are made.

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
    """

    def __init__(self, bucket_name=None, prefix_folder=None, local_path=None):
        """
        bucket_name: S3 bucket name
        prefix_folder: string of folder in S3 bucket where data is stored
        local_path: string of local file path where data is stored
        """
        self.data_types = ["2D-x-core", "2D-y-core", "2D-z-core"]
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
        self.max_values = None
        self.train_patients = None
        self.test_patients = None

        self.prediction_metrics = {"inclusive": {"train": {}, "test": {}}, "exclusive": {"train": {}, "test": {}}}

    def load_dataset(self, patients, brain_cat=True, hist_equal=False, multi_cat=True):
        """
        Loads X and y datasets for the specified patients.
        brain_cat: If True, the brain mask category is kept in the y data.  Else, it is removed.
        hist_equal: If True, histogram equalization is performed on each contrast of X data.  Else, it is not.
        multi_cat: If True, y data is kept in its original state.  Else, all categories are combined into one.
        """
        for patient in patients:
            new_patient = {}
            for data_type in self.data_types:
                new_type = {}
                for data_bin in self.data_bins:
                    key = self.get_key(patient, data_type, data_bin)
                    data = self.load_nifti_file(key)
                    if (data_bin == "X") & hist_equal:
                        num_contrasts = data.shape[-1]
                        if self.max_values is None:
                            self.max_values = np.array([0] * num_contrasts)
                        for ic in range(num_contrasts):
                            max_value = np.max(data[..., ic])
                            if max_value > self.max_values[ic]:
                                self.max_values[ic] = max_value
                    elif data_bin == "y":
                        if not brain_cat:
                            data = data[..., :-1]
                        if not multi_cat:
                            data = np.sum(data, axis=-1, keepdims=True)
                    new_type.update({data_bin: {"key": key, "data": data}})
                new_patient.update({data_type: new_type})
            self.data.update({patient: new_patient})

        if hist_equal:
            self.max_values = self.max_values + np.round(self.max_values / 10)
            for patient in patients:
                for data_type in self.data_types:
                    data = self.data[patient][data_type]["X"]["data"]
                    for ic in range(data.shape[-1]):
                        self.data[patient][data_type]["X"]["data"][..., ic] = exposure.equalize_hist(data[..., ic],
                                                                                            nbins=self.max_values[ic])

    def get_key(self, patient, data_type, data_bin):
        """
        Get the data 'key' for the given patient, data type, and data bin.
        Returns the data 'key'.
        """
        full_path = self.get_full_path(patient, data_type, data_bin)
        if self.local:
            return full_path
        else:
            key = self.bucket.get_key(full_path)
            if not key:
                key = self.bucket.new_key(full_path)
            return key

    def get_full_path(self, patient, data_type, data_bin):
        """
        Get the full folder path for the given patient, data type, and data bin.
        Returns the full path string.
        """
        if self.local:
            return self.local_path + patient + "/" + self.get_file_name(data_type, data_bin)
        else:
            return self.prefix_folder + patient + "/" + self.get_file_name(data_type, data_bin)

    def get_file_name(self, data_type, data_bin):
        """
        Get the file name for the given data type and data bin.
        Returns the file name string.
        """
        return data_bin + "-" + data_type + self.file_type

    def load_nifti_file(self, key):
        """
        Loads a NIfTI file at the given 'key'.  Allows local or remote loading, based on class boolean.
        Returns a numpy array.
        """
        if self.local:
            data = load_nifti_file_local(key)
        else:
            data = load_nifti_file_S3(key)
        return data

    def save_nifti_file(self, data, key):
        """
        Save NIfTI data array to file with the 'key'.  Allows local or remote saving, based on class boolean.
        """
        if self.local:
            save_nifti_file_local(data, key)
        else:
            save_nifti_file_S3(data, key)

    def X(self):
        """
        Returns a numpy array of the X data.
        """
        return self.get_data_subset(self.data.keys(), "X")

    def y(self):
        """
        Returns a numpy array of the y data.
        """
        return self.get_data_subset(self.data.keys(), "y")

    def build_train_test_split(self, train_pct=0.8):
        """
        Creates a random set of patient IDs for selecting a subset of patient data for a train / test split.
        """
        index_train = np.random.choice(len(self.data), int(np.round(len(self.data) * train_pct)), replace=False)
        index_test = [i for i in np.arange(len(self.data)) if i not in index_train]
        patients = list(self.data.keys())
        self.train_patients = [patients[i] for i in index_train]
        self.test_patients = [patients[i] for i in index_test]

        return self.X_train(), self.X_test(), self.y_train(), self.y_test()

    def get_train_test_split(self):
        """
        Returns all data, separated into the train / test split.
        """
        return self.X_train(), self.X_test(), self.y_train(), self.y_test()

    def X_train(self):
        """
        Returns a numpy array of the X train data.
        """
        return self.get_data_subset(self.train_patients, "X")

    def X_test(self):
        """
        Returns a numpy array of the X test data.
        """
        return self.get_data_subset(self.test_patients, "X")

    def y_train(self):
        """
        Returns a numpy array of the y train data.
        """
        return self.get_data_subset(self.train_patients, "y")

    def y_test(self):
        """
        Returns a numpy array of the y test data.
        """
        return self.get_data_subset(self.test_patients, "y")

    def get_data_subset(self, patients, data_bin):
        """
        Creates a numpy array of data in the given data bin for the given set of patients.
        Returns the numpy array of data.
        """
        data_subset = []
        for patient in patients:
            for data_type in self.data_types:
                data_subset.append(self.data[patient][data_type][data_bin]["data"])
        return np.stack(data_subset, axis=0)

    def predict_metrics(self, model, metric, threshold=0.5):
        """
        Calculate the given metrics for train and test predictions for the given model.
        Probability data from model prediction converted into classification data is compared with actual y data.
        The threshold determines the value above which probabilities are considered significant.
        """
        predict_train = model.predict(self.X_train())
        classify_train_inclusive = self.predict_to_classify_inclusive(predict_train, threshold=threshold)
        classify_train_exclusive = self.predict_to_classify_exclusive(predict_train, threshold=threshold)
        y_train = self.y_train()
        predict_test = model.predict(self.X_test())
        classify_test_inclusive = self.predict_to_classify_inclusive(predict_test, threshold=threshold)
        classify_test_exclusive = self.predict_to_classify_exclusive(predict_test, threshold=threshold)
        y_test = self.y_test()

        for ic in range(y_test.shape[-1]):
            self.prediction_metrics["inclusive"]["train"].update(
                {ic: metric(y_train[..., ic].flatten(), classify_train_inclusive[..., ic].flatten())})
            self.prediction_metrics["inclusive"]["test"].update(
                {ic: metric(y_test[..., ic].flatten(), classify_test_inclusive[..., ic].flatten())})
            self.prediction_metrics["exclusive"]["train"].update(
                {ic: metric(y_train[..., ic].flatten(), classify_train_exclusive[..., ic].flatten())})
            self.prediction_metrics["exclusive"]["test"].update(
                {ic: metric(y_test[..., ic].flatten(), classify_test_exclusive[..., ic].flatten())})

    def predict(self, model, model_id):
        """
        Runs the prediction for the given model of all X data.
        Saves the results in NIfTI data files with the model ID tag in the name.
        """
        predict_array = model.predict(self.X())
        predicts = []
        for i in range(predict_array.shape[0]):
            predicts.append(predict_array[i])

        for patient in self.data.keys():
            for data_type in self.data_types:
                key = self.get_key(patient, data_type + "-" + model_id, self.predict_bin)
                data = predicts.pop(0)
                new_bin = {self.predict_bin: {"key": key, "data": data}}
                self.data[patient][data_type].update(new_bin)
                self.save_nifti_file(data, key)

    def predict_to_classify_exclusive(self, predict, threshold=0.5):
        """
        Converts probability data from model prediction into classification data.
        Makes the category call exclusive (only the category with the largest significant probability).
        The threshold determines the value above which probabilities are considered significant.
        Returns the array of classification data.
        """
        num_classes = predict.shape[-1]
        predict[predict < threshold] = 0
        predict = np.lib.pad(predict, ((0, 0), (0, 0), (0, 0), (1, 0)), 'constant', constant_values=0)
        predict = np.argmax(predict, axis=-1)
        classify = []
        for c in range(1, num_classes+1):
            classify.append((predict == c) * 1)
        classify = np.stack(classify, axis=-1)
        return classify

    def predict_to_classify_inclusive(self, predict, threshold=0.5):
        """
        Converts probability data from model prediction into classification data.
        Makes the category call inclusive (any category with significant probability).
        The threshold determines the value above which probabilities are considered significant.
        Returns the array of classification data.
        """
        predict[predict < threshold] = 0
        predict[predict > 0] = 1
        return predict


if __name__ == '__main__':
    """
    Main script for running DataProcessor and DataSet objects.
    local: If True, assumes local source for data files.  Else, assumes S3 location.
    process: If True, runs DataProcessor.  Else, runs DataSet.
    """

    local = True
    process = False

    if local and process:
        dp = DataProcessor(local_path=config.local_path)
        dp.process_data()
    elif local and (not process):
        ds = DataSet(local_path=config.local_path)
        ds.load_dataset(config.pids_of_interest)
        X_train, X_test, y_train, y_test = ds.build_train_test_split()
        X = ds.X()
        y = ds.y()
    elif (not local) and process:
        dp = DataProcessor(config.bucket_name, config.prefix_folder)
        dp.process_data()
    elif (not local) and (not process):
        ds = DataSet(config.bucket_name, config.prefix_folder)
        ds.load_dataset(config.pids_of_interest)
        X_train, X_test, y_train, y_test = ds.build_train_test_split()
        X = ds.X()
        y = ds.y()

    print("Complete.")

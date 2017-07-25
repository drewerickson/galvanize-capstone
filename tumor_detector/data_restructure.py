import config
import os
import shutil
import pandas as pd
import json


def update_data(data, ids, grade=None):
    for id in ids:
        if id not in data:
            data[id] = {config.column_id: id,
                              config.column_grade: grade,
                              config.column_age: None,
                              config.column_survival: None}
        else:
            data[id].update({config.column_grade: grade})


if __name__ == '__main__':
    """
    This script is used for reformatting the data folder structure as it was given to a more consistent structure.
    All subject IDs and additional metadata extracted from the folder structure is stored in the CSV.
    """

    dataset = "valid"

    if dataset == "rain":
        df = pd.read_csv(config.path_train_survival_data, dtype=object)
        hgg_root, hgg_dirs, hgg_files = os.walk(config.path_hgg).__next__()
        lgg_root, lgg_dirs, lgg_files = os.walk(config.path_lgg).__next__()

        data = {}
        for row in df.iterrows():
            data[row[1][config.column_id]] = {config.column_id: row[1][config.column_id],
                                              config.column_age: row[1][config.column_age],
                                              config.column_survival: row[1][config.column_survival]}

        update_data(data, hgg_dirs, grade=config.grade_hgg)
        update_data(data, lgg_dirs, grade=config.grade_lgg)

        data_list = []
        for v in data.values():
            data_list.append(v)

        new_df = pd.DataFrame(data_list, columns=[config.column_id,
                                                  config.column_grade,
                                                  config.column_age,
                                                  config.column_survival])

        new_df.sort_values(by=config.column_id, inplace=True)
        new_df.reset_index(drop=True, inplace=True)
        new_df.to_csv(config.path_train_survival, index=False)

        print(len(new_df))

    elif dataset == "valid":

        # get survival data
        df = pd.read_csv(config.path_valid_in + config.file_survival_in, dtype=object)

        # build a BraTS data dictionary from the survival data
        brats_data = {}
        for row in df.iterrows():
            brats_data[row[1][config.column_id]] = \
                {config.column_id: row[1][config.column_id],
                 config.column_grade: None,
                 config.column_age: row[1][config.column_age],
                 config.column_survival: None}

        # get patient ID directories
        pid_root, pid_dirs, pid_files = os.walk(config.path_valid_in).__next__()

        # add missing patient ID data to the BraTS dictionary
        for pid in pid_dirs:
            if pid not in brats_data:
                brats_data[pid] = \
                    {config.column_id: pid,
                     config.column_grade: None,
                     config.column_age: None,
                     config.column_survival: None}

        # build a new survival data table from the BraTS dictionary
        brats_list = []
        for k, v in brats_data.items():
            brats_list.append(v)
        new_df = pd.DataFrame(brats_list,
                              columns=[config.column_id, config.column_grade, config.column_age, config.column_survival])
        new_df.sort_values(by=config.column_id, inplace=True)
        new_df.reset_index(drop=True, inplace=True)

        # make the output folder
        os.mkdir(config.path_valid_out)

        # save the new table to the output survival file
        new_df.to_csv(config.path_valid_out + config.file_survival_out, index=False)

        # copy the data files into the output folders with the updated name structure
        for valid_dir in pid_dirs:
            os.mkdir(config.path_valid_out + valid_dir)
            valid_sub_root, valid_sub_dirs, valid_sub_files = os.walk(config.path_valid_in + valid_dir).__next__()
            for valid_sub_file in valid_sub_files:
                new_sub_file = valid_sub_file.split("_")[-1]
                shutil.copy(valid_sub_root + "/" + valid_sub_file,
                            config.path_valid_out + valid_dir + "/" + new_sub_file)

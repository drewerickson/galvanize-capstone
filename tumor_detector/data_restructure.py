import config
import os
import pandas as pd


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

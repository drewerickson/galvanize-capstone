import json

column_id = "Brats17ID"
column_grade = "Grade"
column_age = "Age"
column_survival = "Survival"

path_train_survival_data = "/data/brats17/train/survival_data.csv"
path_train_survival = "/data/brats17/train/survival.csv"
path_hgg = "/data/brats17/train/HGG/"
path_lgg = "/data/brats17/train/LGG/"

grade_hgg = "HGG"
grade_lgg = "LGG"

local_path = "/data/brats17_copy/train/"

bucket_name = "dte-brats17"
creds_file = "/data/creds/aws.json"
creds_access_key_name = "access-key"
creds_secret_access_key_name = "secret-access-key"

access_key = None
secret_access_key = None
with open(creds_file) as f:
    json_data = json.load(f)
    access_key = json_data[creds_access_key_name]
    secret_access_key = json_data[creds_secret_access_key_name]

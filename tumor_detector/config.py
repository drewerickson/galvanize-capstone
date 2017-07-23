import json

# CSV variables
column_id = "Brats17ID"
column_grade = "Grade"
column_age = "Age"
column_survival = "Survival"

# Restructuring Variables
path_train_survival_data = "/data/brats17_orig/train/survival_data.csv"
path_train_survival = "/data/brats17_orig/train/survival.csv"
path_hgg = "/data/brats17_orig/train/HGG/"
path_lgg = "/data/brats17_orig/train/LGG/"

grade_hgg = "HGG"
grade_lgg = "LGG"

# Data Variables
local_path = "/data/brats17/train/"
local_out_path = "/data/brats17/train/predict/"
bucket_name = "dte-brats17"
prefix_folder = "train/"
output_folder = "predict/"

# AWS Access Variables
creds_file = "/data/creds/aws.json"  # local
# creds_file = "aws.json"  # EC2
creds_access_key_name = "access-key"
creds_secret_access_key_name = "secret-access-key"

access_key = None
secret_access_key = None
with open(creds_file) as f:
    json_data = json.load(f)
    access_key = json_data[creds_access_key_name]
    secret_access_key = json_data[creds_secret_access_key_name]

# Model Variables
model_folder = "/data/brats17/models"  # local
# model_folder = "models"  # EC2

# Data Selection Variables

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

first_pids = ['Brats17_2013_00_1', 'Brats17_2013_01_1', 'Brats17_2013_02_1', 'Brats17_2013_03_1', 'Brats17_2013_04_1',
              'Brats17_2013_05_1', 'Brats17_2013_06_1', 'Brats17_2013_07_1', 'Brats17_2013_08_1', 'Brats17_2013_09_1',
              'Brats17_2013_10_1', 'Brats17_2013_11_1', 'Brats17_2013_12_1', 'Brats17_2013_13_1', 'Brats17_2013_14_1',
              'Brats17_2013_15_1', 'Brats17_2013_16_1', 'Brats17_2013_17_1', 'Brats17_2013_18_1', 'Brats17_2013_19_1',
              'Brats17_2013_20_1', 'Brats17_2013_21_1', 'Brats17_2013_22_1', 'Brats17_2013_23_1', 'Brats17_2013_24_1',
              'Brats17_2013_25_1', 'Brats17_2013_26_1', 'Brats17_2013_27_1', 'Brats17_2013_28_1', 'Brats17_2013_29_1']

fir_pids_hq = ['Brats17_2013_01_1', 'Brats17_2013_02_1', 'Brats17_2013_03_1', 'Brats17_2013_04_1',
               'Brats17_2013_07_1', 'Brats17_2013_09_1',
              'Brats17_2013_10_1', 'Brats17_2013_11_1', 'Brats17_2013_13_1', 'Brats17_2013_14_1',
              'Brats17_2013_15_1', 'Brats17_2013_16_1', 'Brats17_2013_17_1',
              'Brats17_2013_21_1', 'Brats17_2013_22_1', 'Brats17_2013_24_1',
              'Brats17_2013_27_1', 'Brats17_2013_28_1', 'Brats17_2013_29_1']

cbica_pids = ['Brats17_CBICA_AAB_1', 'Brats17_CBICA_AAG_1', 'Brats17_CBICA_AAL_1', 'Brats17_CBICA_AAP_1',
              'Brats17_CBICA_ABB_1', 'Brats17_CBICA_ABE_1', 'Brats17_CBICA_ABM_1', 'Brats17_CBICA_ABN_1',
              'Brats17_CBICA_ABO_1', 'Brats17_CBICA_ABY_1', 'Brats17_CBICA_ALN_1', 'Brats17_CBICA_ALU_1',
              'Brats17_CBICA_ALX_1', 'Brats17_CBICA_AME_1', 'Brats17_CBICA_AMH_1', 'Brats17_CBICA_ANG_1',
              'Brats17_CBICA_ANI_1', 'Brats17_CBICA_ANP_1', 'Brats17_CBICA_ANZ_1', 'Brats17_CBICA_AOD_1',
              'Brats17_CBICA_AOH_1', 'Brats17_CBICA_AOO_1', 'Brats17_CBICA_AOP_1', 'Brats17_CBICA_AOZ_1',
              'Brats17_CBICA_APR_1', 'Brats17_CBICA_APY_1', 'Brats17_CBICA_APZ_1', 'Brats17_CBICA_AQA_1',
              'Brats17_CBICA_AQD_1', 'Brats17_CBICA_AQG_1', 'Brats17_CBICA_AQJ_1', 'Brats17_CBICA_AQN_1',
              'Brats17_CBICA_AQO_1', 'Brats17_CBICA_AQP_1', 'Brats17_CBICA_AQQ_1', 'Brats17_CBICA_AQR_1',
              'Brats17_CBICA_AQT_1', 'Brats17_CBICA_AQU_1', 'Brats17_CBICA_AQV_1', 'Brats17_CBICA_AQY_1',
              'Brats17_CBICA_AQZ_1', 'Brats17_CBICA_ARF_1', 'Brats17_CBICA_ARW_1', 'Brats17_CBICA_ARZ_1',
              'Brats17_CBICA_ASA_1', 'Brats17_CBICA_ASE_1', 'Brats17_CBICA_ASG_1', 'Brats17_CBICA_ASH_1',
              'Brats17_CBICA_ASK_1', 'Brats17_CBICA_ASN_1', 'Brats17_CBICA_ASO_1', 'Brats17_CBICA_ASU_1',
              'Brats17_CBICA_ASV_1', 'Brats17_CBICA_ASW_1', 'Brats17_CBICA_ASY_1', 'Brats17_CBICA_ATB_1',
              'Brats17_CBICA_ATD_1', 'Brats17_CBICA_ATF_1', 'Brats17_CBICA_ATP_1', 'Brats17_CBICA_ATV_1',
              'Brats17_CBICA_ATX_1', 'Brats17_CBICA_AUN_1', 'Brats17_CBICA_AUQ_1', 'Brats17_CBICA_AUR_1',
              'Brats17_CBICA_AVG_1', 'Brats17_CBICA_AVJ_1', 'Brats17_CBICA_AVV_1', 'Brats17_CBICA_AWG_1',
              'Brats17_CBICA_AWH_1', 'Brats17_CBICA_AWI_1', 'Brats17_CBICA_AXJ_1', 'Brats17_CBICA_AXL_1',
              'Brats17_CBICA_AXM_1', 'Brats17_CBICA_AXN_1', 'Brats17_CBICA_AXO_1', 'Brats17_CBICA_AXQ_1',
              'Brats17_CBICA_AXW_1', 'Brats17_CBICA_AYA_1', 'Brats17_CBICA_AYI_1', 'Brats17_CBICA_AYU_1',
              'Brats17_CBICA_AYW_1', 'Brats17_CBICA_AZD_1', 'Brats17_CBICA_AZH_1', 'Brats17_CBICA_BFB_1',
              'Brats17_CBICA_BFP_1', 'Brats17_CBICA_BHB_1', 'Brats17_CBICA_BHK_1', 'Brats17_CBICA_BHM_1']

cbi_pids_hq = ['Brats17_CBICA_AAB_1', 'Brats17_CBICA_AAG_1', 'Brats17_CBICA_AAL_1', 'Brats17_CBICA_AAP_1',
              'Brats17_CBICA_ABB_1', 'Brats17_CBICA_ABE_1', 'Brats17_CBICA_ABM_1', 'Brats17_CBICA_ABN_1',
              'Brats17_CBICA_ABO_1', 'Brats17_CBICA_ABY_1', 'Brats17_CBICA_ALN_1', 'Brats17_CBICA_ALU_1',
              'Brats17_CBICA_ALX_1', 'Brats17_CBICA_AME_1', 'Brats17_CBICA_AMH_1', 'Brats17_CBICA_ANG_1',
              'Brats17_CBICA_ANI_1', 'Brats17_CBICA_ANP_1', 'Brats17_CBICA_ANZ_1', 'Brats17_CBICA_AOD_1',
              'Brats17_CBICA_AOH_1', 'Brats17_CBICA_AOO_1', 'Brats17_CBICA_AOP_1', 'Brats17_CBICA_AOZ_1',
              'Brats17_CBICA_APR_1', 'Brats17_CBICA_APY_1', 'Brats17_CBICA_APZ_1', 'Brats17_CBICA_AQA_1',
              'Brats17_CBICA_AQD_1', 'Brats17_CBICA_AQG_1', 'Brats17_CBICA_AQJ_1', 'Brats17_CBICA_AQN_1',
              'Brats17_CBICA_AQO_1', 'Brats17_CBICA_AQP_1', 'Brats17_CBICA_AQR_1',
              'Brats17_CBICA_AQT_1', 'Brats17_CBICA_AQU_1', 'Brats17_CBICA_AQV_1', 'Brats17_CBICA_AQY_1',
              'Brats17_CBICA_AQZ_1', 'Brats17_CBICA_ARF_1', 'Brats17_CBICA_ARW_1', 'Brats17_CBICA_ARZ_1',
              'Brats17_CBICA_ASA_1', 'Brats17_CBICA_ASE_1', 'Brats17_CBICA_ASG_1', 'Brats17_CBICA_ASH_1',
              'Brats17_CBICA_ASK_1', 'Brats17_CBICA_ASN_1', 'Brats17_CBICA_ASO_1', 'Brats17_CBICA_ASU_1',
              'Brats17_CBICA_ASV_1', 'Brats17_CBICA_ASW_1', 'Brats17_CBICA_ASY_1', 'Brats17_CBICA_ATB_1',
              'Brats17_CBICA_ATD_1', 'Brats17_CBICA_ATF_1', 'Brats17_CBICA_ATP_1', 'Brats17_CBICA_ATV_1',
              'Brats17_CBICA_ATX_1', 'Brats17_CBICA_AUN_1', 'Brats17_CBICA_AUQ_1', 'Brats17_CBICA_AUR_1',
              'Brats17_CBICA_AVG_1', 'Brats17_CBICA_AVJ_1', 'Brats17_CBICA_AVV_1', 'Brats17_CBICA_AWG_1',
              'Brats17_CBICA_AWH_1', 'Brats17_CBICA_AWI_1', 'Brats17_CBICA_AXJ_1', 'Brats17_CBICA_AXL_1',
              'Brats17_CBICA_AXM_1', 'Brats17_CBICA_AXN_1', 'Brats17_CBICA_AXO_1', 'Brats17_CBICA_AXQ_1',
              'Brats17_CBICA_AXW_1', 'Brats17_CBICA_AYI_1', 'Brats17_CBICA_AYU_1',
              'Brats17_CBICA_AYW_1', 'Brats17_CBICA_AZD_1', 'Brats17_CBICA_AZH_1', 'Brats17_CBICA_BFB_1',
              'Brats17_CBICA_BFP_1', 'Brats17_CBICA_BHB_1', 'Brats17_CBICA_BHK_1', 'Brats17_CBICA_BHM_1']

tcia_pids = ['Brats17_TCIA_101_1', 'Brats17_TCIA_103_1', 'Brats17_TCIA_105_1', 'Brats17_TCIA_109_1',
             'Brats17_TCIA_111_1', 'Brats17_TCIA_113_1', 'Brats17_TCIA_117_1', 'Brats17_TCIA_118_1',
             'Brats17_TCIA_121_1', 'Brats17_TCIA_130_1', 'Brats17_TCIA_131_1', 'Brats17_TCIA_133_1',
             'Brats17_TCIA_135_1', 'Brats17_TCIA_138_1', 'Brats17_TCIA_141_1', 'Brats17_TCIA_147_1',
             'Brats17_TCIA_149_1', 'Brats17_TCIA_150_1', 'Brats17_TCIA_151_1', 'Brats17_TCIA_152_1',
             'Brats17_TCIA_162_1', 'Brats17_TCIA_165_1', 'Brats17_TCIA_167_1', 'Brats17_TCIA_168_1',
             'Brats17_TCIA_171_1', 'Brats17_TCIA_175_1', 'Brats17_TCIA_177_1', 'Brats17_TCIA_179_1',
             'Brats17_TCIA_180_1', 'Brats17_TCIA_184_1', 'Brats17_TCIA_186_1', 'Brats17_TCIA_190_1',
             'Brats17_TCIA_192_1', 'Brats17_TCIA_198_1', 'Brats17_TCIA_199_1', 'Brats17_TCIA_201_1',
             'Brats17_TCIA_202_1', 'Brats17_TCIA_203_1', 'Brats17_TCIA_205_1', 'Brats17_TCIA_208_1',
             'Brats17_TCIA_211_1', 'Brats17_TCIA_218_1', 'Brats17_TCIA_221_1', 'Brats17_TCIA_222_1',
             'Brats17_TCIA_226_1', 'Brats17_TCIA_231_1', 'Brats17_TCIA_234_1', 'Brats17_TCIA_235_1',
             'Brats17_TCIA_241_1', 'Brats17_TCIA_242_1', 'Brats17_TCIA_247_1', 'Brats17_TCIA_249_1',
             'Brats17_TCIA_254_1', 'Brats17_TCIA_255_1', 'Brats17_TCIA_257_1', 'Brats17_TCIA_261_1',
             'Brats17_TCIA_265_1', 'Brats17_TCIA_266_1', 'Brats17_TCIA_274_1', 'Brats17_TCIA_276_1',
             'Brats17_TCIA_277_1', 'Brats17_TCIA_278_1', 'Brats17_TCIA_280_1', 'Brats17_TCIA_282_1',
             'Brats17_TCIA_283_1', 'Brats17_TCIA_290_1', 'Brats17_TCIA_296_1', 'Brats17_TCIA_298_1',
             'Brats17_TCIA_299_1', 'Brats17_TCIA_300_1', 'Brats17_TCIA_307_1', 'Brats17_TCIA_309_1',
             'Brats17_TCIA_310_1', 'Brats17_TCIA_312_1', 'Brats17_TCIA_314_1', 'Brats17_TCIA_319_1',
             'Brats17_TCIA_321_1', 'Brats17_TCIA_322_1', 'Brats17_TCIA_325_1', 'Brats17_TCIA_328_1',
             'Brats17_TCIA_330_1', 'Brats17_TCIA_331_1', 'Brats17_TCIA_332_1', 'Brats17_TCIA_335_1',
             'Brats17_TCIA_338_1', 'Brats17_TCIA_343_1', 'Brats17_TCIA_346_1', 'Brats17_TCIA_351_1',
             'Brats17_TCIA_361_1', 'Brats17_TCIA_368_1', 'Brats17_TCIA_370_1', 'Brats17_TCIA_372_1',
             'Brats17_TCIA_374_1', 'Brats17_TCIA_375_1', 'Brats17_TCIA_377_1', 'Brats17_TCIA_378_1',
             'Brats17_TCIA_387_1', 'Brats17_TCIA_390_1', 'Brats17_TCIA_393_1', 'Brats17_TCIA_394_1',
             'Brats17_TCIA_396_1', 'Brats17_TCIA_401_1', 'Brats17_TCIA_402_1', 'Brats17_TCIA_406_1',
             'Brats17_TCIA_408_1', 'Brats17_TCIA_409_1', 'Brats17_TCIA_410_1', 'Brats17_TCIA_411_1',
             'Brats17_TCIA_412_1', 'Brats17_TCIA_413_1', 'Brats17_TCIA_419_1', 'Brats17_TCIA_420_1',
             'Brats17_TCIA_425_1', 'Brats17_TCIA_428_1', 'Brats17_TCIA_429_1', 'Brats17_TCIA_430_1',
             'Brats17_TCIA_436_1', 'Brats17_TCIA_437_1', 'Brats17_TCIA_442_1', 'Brats17_TCIA_444_1',
             'Brats17_TCIA_448_1', 'Brats17_TCIA_449_1', 'Brats17_TCIA_451_1', 'Brats17_TCIA_455_1',
             'Brats17_TCIA_460_1', 'Brats17_TCIA_462_1', 'Brats17_TCIA_466_1', 'Brats17_TCIA_469_1',
             'Brats17_TCIA_470_1', 'Brats17_TCIA_471_1', 'Brats17_TCIA_473_1', 'Brats17_TCIA_474_1',
             'Brats17_TCIA_478_1', 'Brats17_TCIA_479_1', 'Brats17_TCIA_480_1', 'Brats17_TCIA_490_1',
             'Brats17_TCIA_491_1', 'Brats17_TCIA_493_1', 'Brats17_TCIA_498_1', 'Brats17_TCIA_499_1',
             'Brats17_TCIA_603_1', 'Brats17_TCIA_605_1', 'Brats17_TCIA_606_1', 'Brats17_TCIA_607_1',
             'Brats17_TCIA_608_1', 'Brats17_TCIA_615_1', 'Brats17_TCIA_618_1', 'Brats17_TCIA_620_1',
             'Brats17_TCIA_621_1', 'Brats17_TCIA_623_1', 'Brats17_TCIA_624_1', 'Brats17_TCIA_625_1',
             'Brats17_TCIA_628_1', 'Brats17_TCIA_629_1', 'Brats17_TCIA_630_1', 'Brats17_TCIA_632_1',
             'Brats17_TCIA_633_1', 'Brats17_TCIA_634_1', 'Brats17_TCIA_637_1', 'Brats17_TCIA_639_1',
             'Brats17_TCIA_640_1', 'Brats17_TCIA_642_1', 'Brats17_TCIA_644_1', 'Brats17_TCIA_645_1',
             'Brats17_TCIA_650_1', 'Brats17_TCIA_653_1', 'Brats17_TCIA_654_1']

tci_pids_hq = ['Brats17_TCIA_101_1', 'Brats17_TCIA_103_1', 'Brats17_TCIA_105_1',
               'Brats17_TCIA_118_1',
             'Brats17_TCIA_121_1', 'Brats17_TCIA_130_1', 'Brats17_TCIA_133_1',
             'Brats17_TCIA_138_1',
             'Brats17_TCIA_151_1', 'Brats17_TCIA_152_1',
             'Brats17_TCIA_162_1',
             'Brats17_TCIA_171_1', 'Brats17_TCIA_175_1', 'Brats17_TCIA_177_1',
             'Brats17_TCIA_192_1', 'Brats17_TCIA_198_1', 'Brats17_TCIA_199_1', 'Brats17_TCIA_201_1',
             'Brats17_TCIA_226_1',
             'Brats17_TCIA_241_1', 'Brats17_TCIA_247_1', 'Brats17_TCIA_249_1',
             'Brats17_TCIA_265_1',
             'Brats17_TCIA_298_1',
             'Brats17_TCIA_299_1',
             'Brats17_TCIA_310_1', 'Brats17_TCIA_314_1',
             'Brats17_TCIA_321_1', 'Brats17_TCIA_322_1',
             'Brats17_TCIA_330_1', 'Brats17_TCIA_332_1',
             'Brats17_TCIA_338_1', 'Brats17_TCIA_343_1', 'Brats17_TCIA_346_1', 'Brats17_TCIA_351_1',
             'Brats17_TCIA_368_1',
             'Brats17_TCIA_374_1', 'Brats17_TCIA_375_1', 'Brats17_TCIA_377_1',
             'Brats17_TCIA_409_1', 'Brats17_TCIA_410_1',
             'Brats17_TCIA_419_1',
             'Brats17_TCIA_430_1',
             'Brats17_TCIA_449_1', 'Brats17_TCIA_455_1',
             'Brats17_TCIA_470_1', 'Brats17_TCIA_474_1',
             'Brats17_TCIA_490_1',
             'Brats17_TCIA_493_1', 'Brats17_TCIA_498_1',
             'Brats17_TCIA_603_1', 'Brats17_TCIA_607_1',
             'Brats17_TCIA_634_1']

pids_of_interest = cbi_pids_hq
pids_of_interest.extend(fir_pids_hq)
pids_of_interest.extend(tci_pids_hq)


# test_pids = ['Brats17_TCIA_372_1', 'Brats17_TCIA_637_1', 'Brats17_CBICA_BFB_1', 'Brats17_TCIA_299_1',
#              'Brats17_CBICA_ASH_1', 'Brats17_TCIA_430_1', 'Brats17_TCIA_603_1', 'Brats17_TCIA_310_1',
#              'Brats17_TCIA_109_1', 'Brats17_TCIA_644_1', 'Brats17_CBICA_BHM_1', 'Brats17_TCIA_247_1',
#              'Brats17_TCIA_394_1', 'Brats17_TCIA_321_1', 'Brats17_CBICA_AAL_1', 'Brats17_TCIA_630_1',
#              'Brats17_TCIA_184_1', 'Brats17_CBICA_AYI_1', 'Brats17_TCIA_307_1', 'Brats17_TCIA_396_1',
#              'Brats17_CBICA_BHK_1', 'Brats17_CBICA_AAG_1', 'Brats17_CBICA_AXO_1', 'Brats17_CBICA_AAP_1',
#              'Brats17_CBICA_AVJ_1', 'Brats17_CBICA_ASN_1', 'Brats17_TCIA_437_1', 'Brats17_TCIA_639_1',
#              'Brats17_TCIA_338_1', 'Brats17_TCIA_322_1', 'Brats17_TCIA_470_1', 'Brats17_TCIA_222_1',
#              'Brats17_CBICA_AXW_1', 'Brats17_TCIA_469_1', 'Brats17_TCIA_451_1', 'Brats17_TCIA_420_1',
#              'Brats17_TCIA_261_1', 'Brats17_CBICA_AQY_1', 'Brats17_CBICA_ASE_1', 'Brats17_TCIA_608_1',
#              'Brats17_TCIA_653_1', 'Brats17_CBICA_AZD_1', 'Brats17_TCIA_412_1', 'Brats17_CBICA_ABM_1',
#              'Brats17_CBICA_ATF_1', 'Brats17_CBICA_AAB_1', 'Brats17_TCIA_226_1', 'Brats17_CBICA_AQA_1',
#              'Brats17_TCIA_218_1', 'Brats17_TCIA_624_1', 'Brats17_TCIA_620_1']

test_pids = ['Brats17_TCIA_177_1', 'Brats17_CBICA_AQT_1', 'Brats17_CBICA_ANZ_1', 'Brats17_CBICA_BHM_1',
             'Brats17_CBICA_BFB_1', 'Brats17_CBICA_ARW_1', 'Brats17_TCIA_410_1', 'Brats17_TCIA_101_1',
             'Brats17_2013_09_1', 'Brats17_TCIA_152_1', 'Brats17_TCIA_322_1', 'Brats17_TCIA_171_1',
             'Brats17_CBICA_ATD_1', 'Brats17_TCIA_419_1', 'Brats17_CBICA_ASU_1', 'Brats17_CBICA_APZ_1',
             'Brats17_CBICA_APR_1', 'Brats17_TCIA_338_1', 'Brats17_CBICA_AQR_1', 'Brats17_TCIA_474_1',
             'Brats17_TCIA_121_1', 'Brats17_CBICA_AXW_1', 'Brats17_TCIA_430_1', 'Brats17_2013_28_1',
             'Brats17_2013_22_1', 'Brats17_CBICA_AWI_1', 'Brats17_CBICA_AOP_1', 'Brats17_CBICA_ARZ_1',
             'Brats17_2013_11_1', 'Brats17_TCIA_138_1', 'Brats17_CBICA_AVV_1', 'Brats17_TCIA_130_1']

# pids_of_interest = cbica_pids
# test_pids = ['Brats17_CBICA_AAB_1', 'Brats17_CBICA_AAG_1', 'Brats17_CBICA_AAL_1', 'Brats17_CBICA_AAP_1',
#               'Brats17_CBICA_ABB_1', 'Brats17_CBICA_ABE_1', 'Brats17_CBICA_ABM_1', 'Brats17_CBICA_ABN_1',
#               'Brats17_CBICA_ABO_1', 'Brats17_CBICA_ABY_1', 'Brats17_CBICA_ALN_1', 'Brats17_CBICA_ALU_1',
#               'Brats17_CBICA_ALX_1', 'Brats17_CBICA_AME_1', 'Brats17_CBICA_AMH_1', 'Brats17_CBICA_ANG_1',
#               'Brats17_CBICA_ANI_1', 'Brats17_CBICA_ANP_1']

all_pids = first_pids
all_pids.extend(cbica_pids)
all_pids.extend(tcia_pids)

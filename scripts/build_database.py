import os
from database_tools import BuildDatabase

repo_dir = '/home/cam/Documents/database_tools/'
os.chdir(repo_dir)

worker = BuildDatabase(records_path='data/RECORDS-adults',
                       data_profile_csv='data/sample_count_data.csv',
                       pleth_csv='data/pleth_data.csv',
                       abp_csv='data/abp_data.csv',
                       max_records=200)
worker.run()

import os
from database_tools.tools import BuildDatabase

repo_dir = '/home/camhpj/database_tools/'
os.chdir(repo_dir)

worker = BuildDatabase(records_path='data/RECORDS-adults',
                       data_profile_csv='data/sample_count_data.csv',
                       max_records=10,
                       max_file_size=10,
                       data_dir='physionet.org/files/mimic3wdb/1.0/',
                       output_dir='data/mimic3/')
worker.run()

import os
from database_tools.tools import DataLocator

repo_dir = '/path/to/repository/'
os.chdir(repo_dir)

worker = DataLocator(
    data_dir='path/to/data/dir/',
    mimic3_dir='physionet.org/files/mimic3wdb/1.0/',
)
worker.run()

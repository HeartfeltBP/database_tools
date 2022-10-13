import os
from database_tools.tools import DataLocator

repo_dir = '/media/cam/CAPSTONEDB/database_tools/'
os.chdir(repo_dir)

worker = DataLocator(data_dir='physionet.org/files/mimic3wdb/1.0/')
worker.run()

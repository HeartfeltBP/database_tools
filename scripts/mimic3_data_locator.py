from database_tools.tools import MimicDataLocator
from database_tools.preprocessing.utils import build_data_directory

repo_dir = '/home/cam/Documents/database_tools/'
data_dir = build_data_directory(repo_dir + 'data/', 'mimic3')

worker = MimicDataLocator(
    data_dir=data_dir,
    mimic3_dir='physionet.org/files/mimic3wdb/1.0/',
)
worker.run()

from database_tools.tools import build_data_directory, DataLocator

repo_dir = '/home/cam/Documents/database_tools/'
data_dir = build_data_directory(repo_dir, 'mimic3')

worker = DataLocator(
    data_dir=data_dir,
    mimic3_dir='physionet.org/files/mimic3wdb/1.0/',
)
worker.run()

from database_tools.tools import DataLocator

worker = DataLocator(data_dir='physionet.org/files/mimic3wdb/1.0/')
worker.run()

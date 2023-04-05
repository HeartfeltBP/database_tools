import vitaldb
import pandas as pd
from database_tools.filtering.utils import build_data_directory

repo_dir = '/home/cam/Documents/database_tools/'
data_dir = build_data_directory(repo_dir + 'data/', 'vital')

caseids = vitaldb.find_cases(['ART', 'PLETH'])
df = pd.DataFrame(dict(id=caseids, url=caseids))
df.to_csv(data_dir + 'valid_segs.csv', index=False)

# reads images from 'All_photos'
# looks up in spreadsheet
# moves to folder, appends class name
import pandas as pd
from glob import glob
import shutil

dat = pd.read_csv('Image Classification - Sheet1.csv')

root = 'HX_Ted_2020_NCTC/All_Photos/'
files = glob(root+'*.jpg')

for file in dat['Flooded']:
    try:
        shutil.copyfile(root+file,root+'Water/Water_X_'+file)
    except:
        pass

for file in dat['Not_Flooded']:
    try:
        shutil.copyfile(root+file,root+'No_Water/No_water_X_'+file)
    except:
        pass

for file in dat['Not_sure']:
    try:
        shutil.copyfile(root+file,root+'Not_Sure/Not_sure_X_'+file)
    except:
        pass

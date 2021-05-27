# ## A Transfer Learning Supervised Machine Learning Model for Flooded Road Recognition
# ### Dan Buscombe, Marda Science / USGS
# #### May, 2021; contribution to the COMET "Sunny Day Flooding" project
# adapted from Evan Goldstein's codes at https://github.com/ebgoldstein/NCTrafficCameras

## THIS IS MODEL 3
# 3. supervised "fully coupled" transfer-learned-features--to--class--mapping (this is Evan's model based on mobilenet)
#--> mobilenet feature extraction with distillation head (max pool), and classifying head (dense layer with dropout and kernel regularization),
#--> retrained with data but feature extractor layers keep imagenet weights
#--> highest amount of supervision; no supervised feature extraction, but explicit mapping of those features to classes by iteratively adjusting a model to do so
#--> most amount of parameters, but only classification parameters tuned

# Main changes made to Evan G's original training implementation:
# 1. changed from ipython to python script
# 2. implemented model checkpoint as h5, removed pb/tf-js outputs
# rationale for h5: https://tensorflow.rstudio.com/reference/keras/save_model_hdf5/
# 3. for train/test, switched from generators to arrays. wanted to keep equivalent workflow
# with the unsupervised model I'm making, which needs arrays cos its weird
# 4. I standardize imagery, not normalize
# 5. Used .5 validation split rather than .2
# 6. tested the full ("load model from json / load h5 weights / load image from jpeg / make prediction") workflow

import os
import urllib.request
import datetime
from tensorflow.keras.models import model_from_json
from glob import glob
from skimage.transform import resize
from skimage.io import imread
import numpy as np

##==============================================
# #PREDICTION
##==============================================
def predict_flood(f,FloodCAMML,height_width=224):
    '''
    given file path string, f, and height_width (default=224)
    return 0 for no flood and 1 for flood
    '''
    img = resize(imread(f), (height_width,height_width))
    Orimg = imread(f)
    img_array = np.expand_dims(img,axis=0)

    # do_gradcam_viz(img, FloodCAMML, outfile=f.replace('.jpg','_gradcam.png'))

    return FloodCAMML.predict(img_array, batch_size = 1, verbose = False)

##==============================================
def get_model_load_weights(weights_file):
    '''
    given file path string, weights file, loads keras model and assigns weights
    '''
    json_file = open(weights_file.replace('.h5','.json'), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    FloodCAMML = model_from_json(loaded_model_json)
    #print("Loading weights into model")
    # load weights into new model
    FloodCAMML.load_weights(weights_file)
    return FloodCAMML


# #IMG UTILS
##==============================================
def standardize(img):
    '''
    standardization using adjusted standard deviation,
    then rescales an input dat between mn and mx
    '''
    N = np.shape(img)[0] * np.shape(img)[1]
    s = np.maximum(np.std(img), 1.0/np.sqrt(N))
    m = np.mean(img)
    img = (img - m) / s
    img = rescale(img, 0, 1)
    del m, s, N
    if np.ndim(img)!=3:
        img = np.dstack((img,img,img))

    return img

##==============================================
def rescale(dat,mn,mx):
    '''
    rescales an input dat between mn and mx
    '''
    m = min(dat.flatten())
    M = max(dat.flatten())
    return (mx-mn)*(dat-m)/(M-m)+mn

##==============================================
#The function for the cameras
def GetTrafficCam(URL,camera):

    # retrieve the image
    urllib.request.urlretrieve(URL, "dummy.jpg")

    #determine image name
    ImName = camera + str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")) + '-' + camera + '.jpg'

    #save image
    os.rename('dummy.jpg', ImName)

##==============================================
def LookAtTraffic():

    #print for debug
    print ("looking at traffic...%s" % datetime.datetime.now())

    #cameras:
    GetTrafficCam(Mirlo,'Mirlo')
    GetTrafficCam(Ocracoke,'Ocracoke')
    GetTrafficCam(Hatteras, 'Hatteras')
    GetTrafficCam(Buxton, 'Buxton')
    GetTrafficCam(Canal, 'Canal')
    GetTrafficCam(NewInlet, 'NewInlet')
    GetTrafficCam(NorthDock,'NorthDock')
    GetTrafficCam(SouthDock, 'SouthDock')
    GetTrafficCam(SouthOcracoke, 'SouthOcracoke')

##==============================================
Mirlo = "https://tims.ncdot.gov/TIMS/cameras/viewimage.ashx?id=NC12_MirloBeach.jpg"
Ocracoke = "https://tims.ncdot.gov/TIMS/cameras/viewimage.ashx?id=NC12_OcracokeNorth.jpg"
Hatteras = "https://tims.ncdot.gov/TIMS/cameras/viewimage.ashx?id=NC12_NorthHatterasVillage.jpg"
Buxton = "https://tims.ncdot.gov/TIMS/cameras/viewimage.ashx?id=NC12_Buxton.jpg"
NewInlet = "https://tims.ncdot.gov/TIMS/cameras/viewimage.ashx?id=NC12_NewInlet.jpg"
Canal = "https://tims.ncdot.gov/TIMS/cameras/viewimage.ashx?id=NC12_CanalZone.jpg"
NorthDock = "https://tims.ncdot.gov/TIMS/Cameras/viewimage.ashx?id=Hatteras_Inlet_North_Dock.jpg"
SouthDock = "https://tims.ncdot.gov/TIMS/Cameras/viewimage.ashx?id=Hatteras_Inlet_South_Dock.jpg"
SouthOcracoke = "https://tims.ncdot.gov/tims/cameras/viewimage.ashx?id=Ocracoke_South.jpg"


alpha=0.4
DROPOUT = 0.4#5
NUM_NEURONS = 128 #512
height_width = 224
BS = 16 #12
EPOCHS = 200
PATIENCE = 25
lr = 1e-5 #1e-4

weights_file = 'res/scratch_mv2_bs'+str(BS)+'_drop'+str(DROPOUT)+'_nn'+str(NUM_NEURONS)+'_sz'+str(height_width)+'_lr'+str(lr)+'_v2.h5'


FloodCAMML = get_model_load_weights(weights_file)

#download latest images to test
LookAtTraffic()

print("=================================")
print("=================================")

files = glob('*.jpg')
files = [f for f in files if 'dummy' not in f]
for f in files:
    flood = predict_flood(f,FloodCAMML,height_width)
    if flood>0.5:
        print("%s: flood (prob: %f)" % (f, flood))
    else:
        print("%s: no flood (prob: %f)" % (f, flood))
    print("=================================")

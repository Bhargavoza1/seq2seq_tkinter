#from config.logger import logger
import tensorflow as tf
import os
from shutil import copy2

lagselector = {
    'FRENCH': {
                0: 'http://storage.googleapis.com/download.tensorflow.org/data/fra-eng.zip',
                1: 'fra-eng',
                2: 'fra'},
    'SPANISH': {
                0: 'http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip',
                1: 'spa-eng',
                2: 'spa'}
    }

'''
new_location = os.getcwd()
path_to_zip = tf.keras.utils.get_file(
    lagselector['SPANISH'][1]+'.zip', origin=lagselector['SPANISH'][0],
    cache_subdir=new_location + '/assets',
    extract=True)
path_to_file = os.path.dirname(path_to_zip) + "/"+ lagselector['SPANISH'][1]

try:
    os.mkdir(path_to_file)
except:
    pass
try:
  copy2(new_location + '/assets/'+lagselector['SPANISH'][2]+'.txt', path_to_file)
except:
    pass
'''

path_to_file =''
def loaddata(lagselect = 'FRENCH'):
    global path_to_file
    new_location = os.getcwd()
    path_to_zip = tf.keras.utils.get_file(
        lagselector[lagselect][1] + '.zip', origin=lagselector[lagselect][0],
        cache_subdir=new_location + '/assets',
        extract=True)
    path_to_file = os.path.dirname(path_to_zip) + "/" + lagselector[lagselect][1]

    try:
        os.mkdir(path_to_file)
    except:
        pass
    try:
        copy2(new_location + '/assets/' + lagselector[lagselect][2] + '.txt', path_to_file)
    except:
        pass

    path_to_file = new_location + '/assets/' + lagselector[lagselect][1]+'/'+lagselector[lagselect][2] + '.txt'

    if  os.path.isfile(path_to_file):
       pass
    else:
        raise Exception( path_to_file +'not exists')



def getPreprocessData():
    return  path_to_file





import cv2 # OpenCV
from sklearn.svm import SVC # SVM klasifikator
import os
import joblib
import numpy as np

def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)

def extract_features(img, shape):
    nbins = 9 # broj binova
    cell_size = (8, 8) # broj piksela po celiji
    block_size = (3, 3) # broj celija po bloku

    hog = cv2.HOGDescriptor(_winSize=(shape[1] // cell_size[1] * cell_size[1], 
                                    shape[0] // cell_size[0] * cell_size[0]),
                            _blockSize=(block_size[1] * cell_size[1],
                                        block_size[0] * cell_size[0]),
                            _blockStride=(cell_size[1], cell_size[0]),
                            _cellSize=(cell_size[1], cell_size[0]),
                            _nbins=nbins)

    return hog.compute(img)


def reshape_data(img_feature_set):
    input_data = np.vstack((img_feature_set))
    print(input_data.shape)
    nsamples = 1
    nx, ny = input_data.shape
    return input_data.reshape((nsamples, nx*ny))[0]


def load_model():
    if os.path.isfile('model.joblib'):
        return joblib.load('model.joblib')
    else:
        return SVC(kernel='linear', probability=True)


def main():
    img = load_image('predictTest/neg12.png')
    shape = img.shape
    feature_set = extract_features(img, shape)
    feature_set = np.array(reshape_data(feature_set)).reshape(1, -1)
    clf_svm = load_model()
    result = clf_svm.predict_proba(feature_set)
    print(result)



if __name__ == '__main__':
    main()

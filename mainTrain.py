import os
import numpy as np
import cv2 # OpenCV
from sklearn.svm import SVC # SVM klasifikator
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier # KNN
import matplotlib.pyplot as plt
import joblib

def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)

def display_image(image):
    plt.imshow(image, 'gray')


def load_images(train_dir):
    pos_imgs = []
    neg_imgs = []
    for img_name in os.listdir(train_dir):
        img_path = os.path.join(train_dir, img_name)
        img = load_image(img_path)
        if 'pos' in img_name:
            pos_imgs.append(img)
        elif 'neg' in img_name:
            neg_imgs.append(img)
    print("Positive images #: ", len(pos_imgs))
    print("Negative images #: ", len(neg_imgs))
    return pos_imgs, neg_imgs


def extract_features(pos_imgs, neg_imgs, shape):
    pos_features = []
    neg_features = []
    labels = []

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

    

    for img in pos_imgs:
        pos_features.append(hog.compute(img))

        labels.append(1)

    for img in neg_imgs:
        neg_features.append(hog.compute(img))
        labels.append(0)

    pos_features = np.array(pos_features)
    neg_features = np.array(neg_features)

    x = np.vstack((pos_features, neg_features))
    y = np.array(labels)

    return x, y

def reshape_data(input_data):
    nsamples, nx, ny = input_data.shape
    return input_data.reshape((nsamples, nx*ny))

def save_model(clf_svm):
    joblib.dump(clf_svm, 'model.joblib')

def load_model():
    if os.path.isfile('model.joblib'):
        return joblib.load('model.joblib')
    else:
        return SVC(kernel='linear', probability=True)


def main():
    train_dir = 'testimages' #OVDE PATH OD FOLDERA SA TRENING SKUPOM
    pos_imgs, neg_imgs = load_images(train_dir)


    shape = pos_imgs[0].shape
    x, y = extract_features(pos_imgs, neg_imgs, shape)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    print('Train shape: ', x_train.shape, y_train.shape)
    print('Test shape: ', x_test.shape, y_test.shape)

    x_train = reshape_data(x_train)
    x_test = reshape_data(x_test)

    print('Train shape: ', x_train.shape, y_train.shape)
    print('Test shape: ', x_test.shape, y_test.shape)

    clf_svm = load_model()

    clf_svm.fit(x_train, y_train)
    y_train_pred = clf_svm.predict(x_train)
    y_test_pred = clf_svm.predict(x_test)
    print("Train accuracy: ", accuracy_score(y_train, y_train_pred))
    print("Validation accuracy: ", accuracy_score(y_test, y_test_pred))
    
    save_model(clf_svm)
    

if __name__ == '__main__':
    main()
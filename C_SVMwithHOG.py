# -*- coding: utf-8 -*-

# SUPPORT VECTOR MACHINE WITH HOG
#------------------------------------------------------------------------------
# Objective: To train a SVM with HOG to identify 'thumbs up' and 'thumbs down'.
#------------------------------------------------------------------------------
# Status (last update in 29/07/2020): 
# - Ok!
#------------------------------------------------------------------------------

# Importing libraries ---------------------------------------------------------
import cv2
import numpy as np
import os
import joblib
import pickle
from sklearn import svm 
from skimage.feature import hog
from skimage import exposure

# Reading the images (training) -----------------------------------------------
dirTRAINING = './training'
listTRAINING = [x for x in sorted(os.listdir(dirTRAINING))]
numTRAINING = len(listTRAINING)

training = np.zeros((numTRAINING, 24*32)).astype(np.float64)
for img in range(1, numTRAINING):
    imageNAME = os.path.join(dirTRAINING, listTRAINING[img])
    image = cv2.imread(imageNAME, 0)
    _, hog_image = hog(image, orientations=18, pixels_per_cell=(4, 4),
                       cells_per_block=(1, 1), visualize=True, multichannel=False)
    image = exposure.rescale_intensity(hog_image, in_range=(0, 255)) 
    imageLINE = image.reshape(-1, 24*32).astype(np.float64)
    training[img, :] = imageLINE

# Reading the images (test) ---------------------------------------------------
dirTEST = './test'
listTEST = [x for x in sorted(os.listdir(dirTEST))]
numTEST = len(listTEST)
    
test = np.zeros((numTEST, 24*32)).astype(np.float64)
for img in range(1, numTEST):
    imageNAME = os.path.join(dirTEST, listTEST[img])
    image = cv2.imread(imageNAME, 0)
    _, hog_image = hog(image, orientations=18, pixels_per_cell=(4, 4),
                       cells_per_block=(1, 1), visualize=True, multichannel=False)
    image = exposure.rescale_intensity(hog_image, in_range=(0, 255)) 
    imageLINE = image.reshape(-1, 24*32).astype(np.float64)
    test[img, :] = imageLINE

# Reading the images (generalization) -----------------------------------------
dirGENERALIZATION = './generalization'
listGENERALIZATION = [x for x in sorted(os.listdir(dirGENERALIZATION))]
numGENERALIZATION = len(listGENERALIZATION)

generalization = np.zeros((numGENERALIZATION, 24*32)).astype(np.float64)
for img in range(1, numGENERALIZATION):
    imageNAME = os.path.join(dirGENERALIZATION, listGENERALIZATION[img])
    image = cv2.imread(imageNAME, 0)
    _, hog_image = hog(image, orientations=18, pixels_per_cell=(4, 4),
                       cells_per_block=(1, 1), visualize=True, multichannel=False)
    image = exposure.rescale_intensity(hog_image, in_range=(0, 255)) 
    imageLINE = image.reshape(-1, 24*32).astype(np.float64)
    generalization[img, :] = imageLINE

# Reading the labels ----------------------------------------------------------
trainingCSV = 'trainingLABELS.csv'
trainingLABELS = np.genfromtxt(trainingCSV, delimiter='')

testCSV = 'testLABELS.csv'
testLABELS = np.genfromtxt(testCSV, delimiter='')

generalizationCSV = 'generalizationLABELS.csv'
generalizationLABELS = np.genfromtxt(generalizationCSV, delimiter='')


# Training the SVM ------------------------------------------------------------
clf = svm.SVC()
clf.fit(training, trainingLABELS)


# Veryfing the accuracy of the test -------------------------------------------
result = clf.predict(test)

matches = result == testLABELS
correct = np.count_nonzero(matches)
accuracy = correct*100.0/len(result)

# Veryfing the accuracy of the generalization ---------------------------------
resultGEN = clf.predict(generalization)

matchesGEN = resultGEN == generalizationLABELS
correctGEN = np.count_nonzero(matchesGEN)
accuracyGEN = correctGEN*100.0/len(resultGEN)

# Save frozen model
joblib.dump(clf, './models/svm.sav')
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import svm

target = []
images = []
flat_data = []

DataDir = r'C:\Users\manis\Documents\Programming\New folder (3)\images'
Categories = ['bikes', 'cars', 'dogs']

for category in Categories:
    num = Categories.index(category)
    path = os.path.join(DataDir, category)
    
    for img in os.listdir(path):
        img_arr = imread(os.path.join(path, img))
        img_resized = resize(img_arr, (150,150,3))
        flat_data.append(img_resized.flatten())
        images.append(img_resized)
        target.append(num)

flat_data = np.array(flat_data)
target = np.array(target)
images = np.array(images)

x_train, x_test, y_train, y_test = train_test_split(flat_data, target, test_size=0.3, random_state=109)

param_grid = [
    {'C':[1,10,100,1000], 'kernel':['linear']},
    {'C':[1,10,100,1000], 'gamma':[0.001, 0.0001], 'kernel':['rbf']}
]

svc = svm.SVC(probability=True)
clf = GridSearchCV(svc, param_grid)
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

import pickle
pickle.dump(clf, open('image_model.p', 'wb'))









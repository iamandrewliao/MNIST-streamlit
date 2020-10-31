import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
sns.set(style='white', context='notebook', palette='deep')
np.random.seed(2)

from keras.models import load_model

#setup for KNN (we have the CNN model already saved)
from sklearn import neighbors
from sklearn.model_selection import train_test_split
train = pd.read_csv('C:/Users/Andrew/Documents/coding/data/mnist/train.csv')
y_train = train.label
x_train = train.drop('label', 1)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.1, random_state=2)

import streamlit as st
from PIL import Image, ImageOps
from keras.preprocessing import image
from keras.backend import argmax

#@st.cache(allow_output_mutation=True) #caches functions: https://docs.streamlit.io/en/stable/api.html#streamlit.cache
def getclf(clf_name):
  clf = None
  if clf_name == 'KNN':
    clf = neighbors.KNeighborsClassifier(n_neighbors=3).fit(x_train, y_train)
  elif clf_name == 'CNN':
    clf = load_model('C:/Users/Andrew/Downloads/mnistcnn.h5')
  return clf

def predict(image_data, clf):
	size = (28, 28)
	img = ImageOps.fit(image_data, size, Image.ANTIALIAS)
	img = img.convert('L')
	img = np.asarray(img)
	img = img/255.0
	if classifier == 'CNN':
  		img = img.reshape(-1,28,28,1)
	elif classifier == 'KNN':
  		img = img.reshape(1, 28*28)
	prediction = clf.predict(img)
	return prediction

st.title('MNIST: CNN vs. KNN')

classifier = st.sidebar.selectbox('Classifers', ("CNN", "KNN"))

clf = getclf(classifier)
st.write(f'Classifier = {classifier}')

st.set_option('deprecation.showfileUploaderEncoding', False) #ignores a warning
file = st.file_uploader('Upload image of a number from 1 to 9', type=['jpg', 'png'])

if file is None:
	st.text("Please upload an image file")
else:
	image = Image.open(file)
	st.image(image, use_column_width=True)
	prediction = predict(image, clf)
	if classifier == 'CNN':
		#class_names = [str(i) for i in range(10)]
		#output = f'Prediction: {class_names[np.argmax(prediction)]}'
		output = f'Prediction: {np.argmax(prediction)}'
	elif classifier == 'KNN':
		output = f'Prediction: {prediction[0]}'
	st.success(output)
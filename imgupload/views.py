from django.shortcuts import render
from django.core.files.storage import FileSystemStorage

import tensorflow as tf
import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras_preprocessing import image
import numpy as np

model_graph = tf.Graph()
with model_graph.as_default():
    sess = tf.Session()
    with sess.as_default():
        model=load_model('./models/model_keras.h5')

def home(request):

    return render(request,'home.html',{})

def predictimage(request):
    print(request)
    print(request.POST.dict())
    fobj=request.FILES['testimg']
    fl=FileSystemStorage()
    filepath = fl.save(fobj.name,fobj)
    filepath = fl.url(filepath)
    testimg='.'+filepath
    print(testimg)
    img = image.load_img(testimg,target_size=(150,150))
    x=np.array(img)
    print(x)
    x= np.reshape(x,(1,150,150,3))
    with model_graph.as_default():
        with sess.as_default():
            predict = model.predict(x)
            print("my prediction.................")
            if predict>0.5:
                print("Dog")
                name="Dog"
            else:
                print("Cat")
                name="Cat"
    
    context = {'filepath':filepath , 'name':name}
    return render(request,'home.html',context)
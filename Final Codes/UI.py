# -*- coding: utf-8 -*-
"""
Created on Fri May 13 19:50:47 2022

@author: naveens
"""

import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from tkinter.filedialog import askopenfile
from PIL import Image, ImageTk
import pickle
import numpy as np 
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from skimage.feature import greycomatrix, greycoprops
from skimage.measure import shannon_entropy

my_w = tk.Tk()
my_w.geometry("850x850")  # Size of the window 
my_w.title('Naturoleaf')
my_font1=('times', 18, 'bold')

frame0=tk.Frame(my_w,width=700,height =100,highlightbackground='white', highlightthicknes=3)
frame0.grid(row=1,columnspan=4)
Label(frame0, text= "Naturoleaf - Medicinal Plant Detection", font=('Mistral 18 bold')).place(x=80,y=40)

frame1=tk.Frame(my_w,width=200,height =100,highlightbackground='red', highlightthicknes=3)
frame1.grid (row=2, column=1) 
b1 = tk.Button(my_w, text='Capture', 
   width=10,command = lambda:camera_input())
b1.grid(row=2,column=1) 

def upload_file():
    global img
    global image_path
    f_types = [('Jpg Files', '*.jpg')]
    filename = filedialog.askopenfilename(filetypes=f_types)
    img = ImageTk.PhotoImage(file=filename)
    frame = Frame(my_w, width=600, height=400)
    frame.grid()
    frame.place(anchor='center', relx=0.5, rely=0.6)
    label = Label(frame, image = img)
    label.grid()
    print(type(img))
    image_path = filename

frame2=tk.Frame(my_w,width=200,height =100,highlightbackground='red', highlightthicknes=3)
frame2.grid (row=2, column=2) 
b3 = tk.Button(my_w, text='Upload', 
   width=10,command = lambda:upload_file())
b3.grid(row=2,column=2) 

def camera_input():
    global img
    global image_path
    cap=cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        cv2.imshow('Video Feed', frame)
        if cv2.waitKey(1) & 0xFF ==ord('q'):
            cv2.imwrite('Webcam1.jpg',frame)
            break
    cap.release()
    cv2.destroyAllWindows()
    
    
    ##CLAHE 
    imgg=cv2.imread('Webcam1.jpg', 1)
    lab_img=cv2.cvtColor(imgg,cv2.COLOR_BGR2LAB)
    l,a,b=cv2.split(lab_img)
    clahe=cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    clahe_img=clahe.apply(l)
    updated_lab_img2=cv2.merge((clahe_img,a,b))
    CLAHE_img=cv2.cvtColor(updated_lab_img2,cv2.COLOR_LAB2BGR)
    cv2.imshow('CLAHE',CLAHE_img)
    cv2.imwrite('CLAHE.jpg',CLAHE_img)
    cv2.waitKey()    
    cv2.destroyAllWindows()
    filename = 'CLAHE.jpg'
    image_path = filename
    
    



frame4=tk.Frame(my_w,width=200,height =100,highlightbackground='red', highlightthicknes=3)
frame4.grid (row=2, column=3) 
b5 = tk.Button(my_w, text='Predict', 
   width=10,command = lambda:predict())
b5.grid(row=2,column=3) 



def open_popup(text):
   top= Toplevel(my_w)
   top.geometry("600x150")
   top.title("Prediction Result")
   Label(top, text= text, font=('Mistral 18 bold')).place(x=80,y=40)


def predict():
    SIZE = 256
    lgb_model = pickle.load(open("finalized_model(lgbm).sav", "rb"))
    print(image_path)
    img1 = cv2.imread(image_path,0)
    
    img1 = cv2.resize(img1,(SIZE,SIZE))
    #plt.imshow(img)
    
    #Extract features and reshape to right dimensions
    input_img = np.expand_dims(img1, axis=0) #Expand dims so the input is (num images, x, y, c)
    input_img_features=feature_extractor(input_img)
    input_img_features = np.expand_dims(input_img_features, axis=0)
    input_img_for_RF = np.reshape(input_img_features, (input_img.shape[0], -1))
    #Predict
    img_prediction = lgb_model.predict(input_img_for_RF)
    img_prediction=np.argmax(img_prediction, axis=1)
    #inverse-transform of labels
    labels = ["Besella Alba (Basale)", "Carissa Carandas (Karanda)","Ficus Religiosa(Peepal Tree)","Jasminum (Jasmine)","Magnifera Indica (Mango)", "Mentha (Mint)", "Moringa Oleifera (Drunmstick)", "Ocimim Tenuiflorum (Tulsi)", "Psidium Guajava (Guava)"]
    idx = img_prediction[0]
    print("Predicted leaf: "+str(labels[idx]))
    open_popup(labels[idx])
    
#-----------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------
#---------DONT CHANGE ANYTHING BELOW------------------------------------------------


def feature_extractor(dataset):
    image_dataset = pd.DataFrame()
    for image in range(dataset.shape[0]):  #iterate through each file 
        #print(image)
        
        df = pd.DataFrame()  #Temporary data frame to capture information for each loop.
        #Reset dataframe to blank after each loop.
        
        img = dataset[image, :,:]
    ################################################################
    #START ADDING DATA TO THE DATAFRAME
  
                
         #Full image
        #GLCM = greycomatrix(img, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4])
        GLCM = greycomatrix(img, [1], [0])       
        GLCM_Energy = greycoprops(GLCM, 'energy')[0]
        df['Energy'] = GLCM_Energy
        GLCM_corr = greycoprops(GLCM, 'correlation')[0]
        df['Corr'] = GLCM_corr       
        GLCM_diss = greycoprops(GLCM, 'dissimilarity')[0]
        df['Diss_sim'] = GLCM_diss       
        GLCM_hom = greycoprops(GLCM, 'homogeneity')[0]
        df['Homogen'] = GLCM_hom       
        GLCM_contr = greycoprops(GLCM, 'contrast')[0]
        df['Contrast'] = GLCM_contr


        GLCM2 = greycomatrix(img, [3], [0])       
        GLCM_Energy2 = greycoprops(GLCM2, 'energy')[0]
        df['Energy2'] = GLCM_Energy2
        GLCM_corr2 = greycoprops(GLCM2, 'correlation')[0]
        df['Corr2'] = GLCM_corr2       
        GLCM_diss2 = greycoprops(GLCM2, 'dissimilarity')[0]
        df['Diss_sim2'] = GLCM_diss2       
        GLCM_hom2 = greycoprops(GLCM2, 'homogeneity')[0]
        df['Homogen2'] = GLCM_hom2       
        GLCM_contr2 = greycoprops(GLCM2, 'contrast')[0]
        df['Contrast2'] = GLCM_contr2

        GLCM3 = greycomatrix(img, [5], [0])       
        GLCM_Energy3 = greycoprops(GLCM3, 'energy')[0]
        df['Energy3'] = GLCM_Energy3
        GLCM_corr3 = greycoprops(GLCM3, 'correlation')[0]
        df['Corr3'] = GLCM_corr3       
        GLCM_diss3 = greycoprops(GLCM3, 'dissimilarity')[0]
        df['Diss_sim3'] = GLCM_diss3       
        GLCM_hom3 = greycoprops(GLCM3, 'homogeneity')[0]
        df['Homogen3'] = GLCM_hom3       
        GLCM_contr3 = greycoprops(GLCM3, 'contrast')[0]
        df['Contrast3'] = GLCM_contr3

        GLCM4 = greycomatrix(img, [0], [np.pi/4])       
        GLCM_Energy4 = greycoprops(GLCM4, 'energy')[0]
        df['Energy4'] = GLCM_Energy4
        GLCM_corr4 = greycoprops(GLCM4, 'correlation')[0]
        df['Corr4'] = GLCM_corr4       
        GLCM_diss4 = greycoprops(GLCM4, 'dissimilarity')[0]
        df['Diss_sim4'] = GLCM_diss4       
        GLCM_hom4 = greycoprops(GLCM4, 'homogeneity')[0]
        df['Homogen4'] = GLCM_hom4       
        GLCM_contr4 = greycoprops(GLCM4, 'contrast')[0]
        df['Contrast4'] = GLCM_contr4
        
        GLCM5 = greycomatrix(img, [0], [np.pi/2])       
        GLCM_Energy5 = greycoprops(GLCM5, 'energy')[0]
        df['Energy5'] = GLCM_Energy5
        GLCM_corr5 = greycoprops(GLCM5, 'correlation')[0]
        df['Corr5'] = GLCM_corr5       
        GLCM_diss5 = greycoprops(GLCM5, 'dissimilarity')[0]
        df['Diss_sim5'] = GLCM_diss5       
        GLCM_hom5 = greycoprops(GLCM5, 'homogeneity')[0]
        df['Homogen5'] = GLCM_hom5       
        GLCM_contr5 = greycoprops(GLCM5, 'contrast')[0]
        df['Contrast5'] = GLCM_contr5
        
        #Add more filters as needed
        entropy = shannon_entropy(img)
        df['Entropy'] = entropy

        
        #Append features from current image to the dataset
        image_dataset = image_dataset.append(df)
        
    return image_dataset

my_w.mainloop()  # Keep the window open
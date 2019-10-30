'''
Copyright (c) 2019 Intel Corporation.

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

import os

import logging
logging.basicConfig(
	filename='app.log', 
	filemode='w', 
	format='%(name)s - %(levelname)s - %(message)s',
	level=logging.INFO)

import sys
sys.path.append("C:\\Program Files (x86)\\IntelSWTools\\openvino\\python\\python3.6")
sys.path.append("C:\\Program Files (x86)\\IntelSWTools\\openvino\\inference_engine\\bin\\intel64\\Release")

import tkinter as tk
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
from tkinter import *
from tkinter import messagebox 

import numpy as np
import cv2
from PIL import ImageTk, Image
import PIL.Image, PIL.ImageTk
import time
from openvino.inference_engine import IENetwork, IEPlugin

from keras.applications.inception_v3 import preprocess_input

home = tk.Tk()
home.geometry("900x800")
home.title("Inference App")


def file_browse(filename):
	#Browse to the Model XML file. 
	if(filename == "XML"):	
		fname = askopenfilename(initialdir=os.getcwd(),title = "Select XML file",filetypes=(("XML Files", "*.xml"),                                           
	                                           ("All files", "*.*") ))
		try:
			xml_entry.delete(0,'end')
			#entry.insert(0,os.path.basename(fname))
			xml_entry.insert(0,fname)			
		except:
			print('Exception')
	#Browse to the label file
	elif(filename == "BIN"):
		fname = askopenfilename(initialdir=os.getcwd(),title = "Select Labels file",filetypes=(("text Files", "*.txt"),                                           
	                                           ("All files", "*.*") ))
		try:
			label_entry.delete(0,'end')			
			label_entry.insert(0,fname)
			
		except:
			print('Exception')
	#Browse to the Image files or the Video files that needs to be Inferenced on
	elif(filename == "Media"):
		media_option = strCapture.get()
		if(media_option == "Image"):
			fname = askopenfilename(filetypes=(("jpeg Files", "*.jpg *.jpeg"),                                   
	                                           ("All files", "*.*") ))
		elif(media_option == "Video"):
			fname = askopenfilename(filetypes=(("MP4", "*.mp4"),("AVI", "*.avi"),                                          
	                                           ("All files", "*.*") ))
		else:
			pass


		if(not fname):
			return

		try:
			
			image_entry.delete(0,'end')			
			image_entry.insert(0,fname)
			#Set the selected image to the label of the GUI 
			if(media_option == "Image"):			
			
				img = cv2.imread(fname)
				fscale = image_scale.get()
				float_fscale = 1/float(fscale)
				#Scaling the image to the scale factor set in the GUI. Default here is 3
				res = cv2.resize(img, (int(img.shape[1]/float(float_fscale)), int(img.shape[0]/float(float_fscale))))
				res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)			
				img2 = Image.fromarray(res, 'RGB')
				img = ImageTk.PhotoImage(img2)			
				label = tk.Label(image_frame,image = img)
				label.image = img
				label.place(relwidth=1, relheight=1)
			else:
				pass			
			
		except BaseException as e:
			msg = messagebox.showinfo("Inference", "An exception is encountered. "+str(e))
			
	else:
		pass	
		
	return




################################################################################
# 
# Inference function	
def Inference():

	try:

		#Get the Model XML file from the XML entry of the GUI
		model_xml = xml_entry.get()
		if(not model_xml):
			msg = messagebox.showinfo("Open Vino Inference", "Please browse and select the model XML file")
			return

		#Set the Model file as per the naming convention of the XML. Assumption here is that the XML and BIN files are the same naming format
		model_bin = os.path.splitext(model_xml)[0] + ".bin"
		
		#Get and set the label file path
		labels_filepath = label_entry.get()
		
		if(not labels_filepath):
			msg = messagebox.showinfo("Inference", "Please browse and select the labels file")
			return
		#Get the device option from one of the three options(CPU, GPU or Movidius)
		device_option = strDevice.get()		

		if(device_option== "Movidius"):
			device_option = "MYRIAD"

		#Get the feed option here as the input. This is from one of the three options(Image, Video or Movidius)
		feed_option = strCapture.get()
		fileext = os.path.splitext(image_entry.get())

		#Valiating ths user input for the selected image. If it is empty promt to select the Image again
		if((feed_option == "Image") and (not image_entry.get())):
			msg = messagebox.showinfo("Inference", "Please browse and select the Image files")
			return
		#Valiating ths user input for the selected Video. If it is empty promt to select the Video file again
		if((feed_option == "Video") and (not image_entry.get())):
			msg = messagebox.showinfo("Inference", "Please browse and select the Video files")
			return

		#Valiating ths selected image if it is a JPG image
		if(feed_option == "Image" and (fileext[1].lower() != ".jpg" and  fileext[1].lower() != ".jpeg" )):
			msg = messagebox.showinfo("Inference", "Please select valid Image files. Jpeg files are supported")
			return
		#Valiating ths selected Video if it is a .MP4,.MPEG or .AVI format 
		if(feed_option == "Video" and (fileext[1].lower() != ".mp4" and  fileext[1].lower() != ".avi" and  fileext[1].lower() != ".mpeg" )):
			msg = messagebox.showinfo("Inference", "Please enter valid Video files. MP4,AVI or MPEG files are supported")
			return
				
		image_list = []
		#if the Option to Scan the full folder is selected
		if(scanFolder.get()):
			#For video option only one video that is selected is appended. Does not support multiple videos if exist in that folder
			if(feed_option == "Video"):
				image_list.append(image_entry.get())
			#For images when Scan folder is selected, it browses through all the JPG files in the folder and appends to the list to do Inference
			elif(feed_option == "Image"):			
				full_folder = os.path.dirname(image_entry.get())
				for file_name in os.listdir(full_folder):
					if file_name.endswith(".jpg") or file_name.endswith(".jpeg"):
						image_list.append(os.path.join(full_folder, file_name))
					else:
						continue				
			else:
				pass
		else:
			image_list.append(image_entry.get())				
		# Plugin initialization for specified device
		plugin = IEPlugin(device=device_option)
		# Read IR		
		net = IENetwork(model=model_xml, weights=model_bin)		
		
		input_blob, out_blob = next(iter(net.inputs)), next(iter(net.outputs))
		net.batch_size = 1

	################################################################################
	    #Loading the labels file
		with open(labels_filepath, 'r') as f: labels_map = [x.split(sep=' ', maxsplit=1)[-1].strip() for x in f]
		
		# Read and pre-process input image
		n, c, h, w = net.inputs[input_blob].shape
		exec_net = plugin.load(network=net)
		
		#if option selected is image
		if(feed_option == "Image"):
			#Scan thorugh the Image folder list and starting Inference on the Images
			for i in range(len(image_list)):
				tm = time.time()
				cap = cv2.imread(image_list[i])
				frame_ = cap
				frame_ = preprocess_input(frame_)
				res = exec_net.infer(inputs={input_blob: [cv2.resize(frame_, (w, h)).transpose((2, 0, 1))]})
				res = res[out_blob]
				clslbl = []
				#Calculating the threshold probabablity
				for i, probs in enumerate(res):
					top_ind = np.argsort(np.squeeze(probs))[-10:][::-1]
					for id in top_ind:
						clslbl.append("{} ({:.2f}%)".format(labels_map[id], 100*probs[id]))
				frame_ = cv2.resize(frame_, (800, 460))
				#Showing the Image, threshold and the Inferences per second details on the frame
				txt = ('[%02d INF/S] Prediction: ' % (1/(time.time() - tm))) + clslbl[0]		
				txtlabel.config(text = txt)

				#If scanning thorugh the complete folder, showing the Image slide show,shows the Inference results along with the Threshold details
				#The image changes every 2 seconds in the slide show and one can quit by pressing the q button
				if(scanFolder.get()):
					final_txt = "Press \\'q\\'  to Quit"	
					cv2.putText(frame_, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.80, (0, 255, 0), 2)
					cv2.putText(frame_, final_txt, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, .80, (0, 255, 0), 2)								
					cv2.imshow('Inference', frame_)
					if cv2.waitKey(2000) & 0xFF == ord('q'): break
		#If Camera option is selected, starts the live camera feed			
		elif(feed_option == "Camera"):
			cap = cv2.VideoCapture(0)		
			while TRUE:
				tm = time.time()
				ret, frame_ = cap.read()
				frame_ = preprocess_input(frame_)
				res = exec_net.infer(inputs={input_blob: [cv2.resize(frame_, (w, h)).transpose((2, 0, 1))]})
				res = res[out_blob]
				clslbl = []
				#Calculating the threshold probabablity
				for i, probs in enumerate(res):
					top_ind = np.argsort(np.squeeze(probs))[-10:][::-1]
					for id in top_ind:
						clslbl.append("{} ({:.2f}%)".format(labels_map[id], 100*probs[id]))
				frame_ = cv2.resize(frame_, (820, 460))
				txt = ('[%02d FPS] Prediction: ' % (1/(time.time() - tm))) + clslbl[0]
				final_txt = "Press \\'q\\'  to Quit"
				cv2.putText(frame_, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,0.80, (0, 255, 0), 2)
				cv2.putText(frame_, final_txt, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, .80, (0, 255, 0), 2)
				#showing the Camera live feed and shows the Inference results in a separate frame along with the Threshold details and the FPS
				# one can quit by pressing the q button
				cv2.imshow('Inference', frame_)	
				if cv2.waitKey(1) & 0xFF == ord('q'): break
		else:
			#If Video is selected as the capture option, loads the selected video and starts inference process
			cap = cv2.VideoCapture(image_list[0])		
			while(cap.isOpened()):
				tm = time.time()
				ret, frame_ = cap.read()
				frame_ = preprocess_input(frame_)
				res = exec_net.infer(inputs={input_blob: [cv2.resize(frame_, (w, h)).transpose((2, 0, 1))]})
				res = res[out_blob]
				clslbl = []
				for i, probs in enumerate(res):
					top_ind = np.argsort(np.squeeze(probs))[-10:][::-1]
					for id in top_ind:
						clslbl.append("{} ({:.2f}%)".format(labels_map[id], 100*probs[id]))
				frame_ = cv2.resize(frame_, (820, 460))
				txt = ('[%02d FPS] Prediction: ' % (1/(time.time() - tm))) + clslbl[0]
				final_txt = "Press \\'q\\'  to Quit"
				cv2.putText(frame_, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,0.80, (0, 255, 0), 2)
				cv2.putText(frame_, final_txt, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, .80, (0, 255, 0), 2)
				#Loads the video file in a separate frame and shows the inference results like Threshold details and the FPS
				# one can quit by pressing the q button
				cv2.imshow('Inference', frame_)	
				if cv2.waitKey(1) & 0xFF == ord('q'): break			
		
		#cap.release()
		cv2.destroyAllWindows()
		del exec_net, plugin, net
	except BaseException as e:
		msg = messagebox.showinfo("Inference", "An exception is encountered. "+str(e))
		

	
################################################################################

#GUI coding goes here
#setting the background color
bg_label = tk.Label(home, bg='#50514F')
bg_label.place(relwidth=1, relheight=1)

#Upper Widget frame with Entries andButtons
widgetframe = tk.Frame(home, bg='#0071C5', bd=10)
widgetframe.place(relx=0.5, rely=0.01, relwidth=0.75, relheight=0.25, anchor='n')

#XML entry to provide the path to the model XML file 
xml_entry = tk.Entry(widgetframe,bg='#A3D5FF', font="Verdana 10 bold")
xml_entry.place(relwidth=0.65, rely = 0.05,relheight=0.15)

#Button to browse the path to the model XML file
xml_button = tk.Button(widgetframe,bg='#A3D5FF', text="Browse XML", font="Verdana 10 bold", command=lambda: file_browse("XML"))
xml_button.place(relx=0.7,rely=.05, relheight=0.15, relwidth=0.3)

#label entry to provide the path to the Labels file
label_entry = tk.Entry(widgetframe,bg='#A3D5FF', font="Verdana 10 bold")

label_entry.place(rely = 0.2,relwidth=0.65, relheight=0.15)
#Button to browse the path to the labels file
label_button = tk.Button(widgetframe,bg='#A3D5FF', text="Browse Label", font="Verdana 10 bold", command=lambda: file_browse("BIN"))
label_button.place(relx=0.7,rely=.2, relheight=0.15, relwidth=0.3)

#Image entry to provide the path to the Image or Video file
image_entry = tk.Entry(widgetframe,bg='#A3D5FF', font="Verdana 10 bold")
image_entry.place(rely = 0.50,relwidth=0.65, relheight=0.15)

#Button to browse to the Image or Video file path 
image_button = tk.Button(widgetframe,bg='#A3D5FF', text="Browse Media", font="Verdana 10 bold", command=lambda: file_browse("Media"))
image_button.place(relx=0.7,rely=.50, relheight=0.15, relwidth=0.3)
#Check box for "Scan Folder"
scanFolder = IntVar()
multiImagebtn = tk.Checkbutton(widgetframe,text = "Scan Folder", variable=scanFolder,bg='#A3D5FF',font = "Verdana 10 bold")
multiImagebtn.place(rely = 0.67)
#This option is for setting the scale of the Image. Default is 3
image_scale = tk.Entry(widgetframe,bg='#A3D5FF', font="Verdana 11 bold")

image_scale.place(rely = 0.35,relwidth=0.25, relheight=0.15)
image_scale.insert(0,3)

scale_label = tk.Label(widgetframe,bg='#9AA0A8', text="Image scale ", justify = tk.LEFT, font="Verdana 11 bold")

scale_label.place(relx=0.7,rely=.35, relheight=0.15, relwidth=0.3)
#Inference button to start on teh Inference process
inf_button = tk.Button(widgetframe,bg='#A3D5FF', text="Inference", font="Verdana 10 bold", command=lambda: Inference())
inf_button.place(relx=0.385,rely=.79, relheight=0.25, relwidth=0.2)


#Radio buttons to select one of the CPU, GPU or Movidius options
strDevice = StringVar(value="CPU")
strCapture = StringVar(value="Image")
btnCPU = Radiobutton(widgetframe, text = "CPU", variable = strDevice, value = "CPU",bg='#A3D5FF',font = "Verdana 10 bold")
btnCPU.pack(anchor =S,side = LEFT)

btnGPU = Radiobutton(widgetframe, text = "GPU", variable = strDevice, value = "GPU",bg='#A3D5FF',font = "Verdana 10 bold")
btnGPU.pack(anchor = S,side = LEFT)

btnMod = Radiobutton(widgetframe, text = "Movidius", variable = strDevice, value = "Movidius",bg='#A3D5FF',font = "Verdana 10 bold")
btnMod.pack(anchor = S,side = LEFT)

#Radio buttons to select one of the capture otpions like Image, Video or Camera
btnCam = Radiobutton(widgetframe, text = "Camera", variable = strCapture, value = "Camera",bg='#A3D5FF',font = "Verdana 10 bold")
btnCam.pack(anchor =S,side = RIGHT)

btnImg = Radiobutton(widgetframe, text = "Image", variable = strCapture, value = "Image",bg='#A3D5FF',font = "Verdana 10 bold")
btnImg.pack(anchor = S,side = RIGHT)

btnVid = Radiobutton(widgetframe, text = "Video", variable = strCapture, value = "Video",bg='#A3D5FF',font = "Verdana 10 bold")
btnVid.pack(anchor = S,side = RIGHT)

#Middle frame that loads the selected Image
image_frame = tk.Frame(home, bg='#0071C5', bd=10)
image_frame.place(relx=0.5, rely=0.27, relwidth=0.75, relheight=0.58, anchor='n')

#Label in the Middle frame that loads the selected Image
label = tk.Label(image_frame,bg='#A3D5FF')
label.place(relwidth=1, relheight=1)

#Lower frame that shows the Inference results in the below text Label
text_frame = tk.Frame(home,bg="#D9F0FF", bd=10)
text_frame.place(relx=0.5, rely=0.86, relwidth=0.75, relheight=0.10, anchor='n')


txtlabel = tk.Label(text_frame,bg='#D9F0FF',text= "",justify=tk.LEFT, padx = 10,font = "Verdana 15 bold")
txtlabel.place(relwidth=1, relheight=1)

home.mainloop()
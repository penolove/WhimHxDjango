# import the necessary packages
from django.shortcuts import get_object_or_404,render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.conf import settings
import numpy as np
import urllib
import json
import cv2
import glob
import pickle 

from .forms import DocumentForm


with open(settings.BASE_DIR+'/Whimh_model.dkl','rb') as input:
	model=pickle.load(input)

cascPath = settings.BASE_DIR+"/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

@csrf_exempt
def detect(request):
	# initialize the data dictionary to be returned by the request
	data = {"success": False}
 	count='0'
 	form = DocumentForm() 

	# check to see if this is a post request
	if request.method == "POST":
		# check to see if an image was uploaded
		if request.FILES.get("docfile", None) is not None:
			# grab the uploaded image
			count = _grab_image(stream=request.FILES["docfile"])
			print "streams,streams,streams,streams"
			return render(request, 'whimh2/index.html',{'imgid':str(count), 'form': form})
		# otherwise, assume that a URL was passed in
		else:
			# grab the URL from the request
			url = request.POST.get("url", None)
 
			# if the URL is None, then return an error
			if url is None:
				data["error"] = "No URL provided."
				return render(request, 'whimh2/index.html',{'imgid':str(count), 'form': form})
 			print "urls"
			# load the image and convert	
			count = _grab_image(url=url)
			return render(request, 'whimh2/index.html',{'imgid':str(count), 'form': form})
		
	print "not post not post not post not post not post"
	return render(request, 'whimh2/index.html',{'imgid':str(count), 'form': form})
 
def _grab_image(path=None, stream=None, url=None):
	# if the path is not None, then load the image from disk
	if path is not None:
		image = cv2.imread(path)
 
	# otherwise, the image does not reside on disk
	else:	
		# if the URL is not None, then download the image
		if url is not None:
			resp = urllib.urlopen(url)
			data = resp.read()
 
		# if the stream is not None, then the image has been uploaded
		elif stream is not None:
			data = stream.read()
 
		# convert the image to a NumPy array and then read it into
		# OpenCV format
		image = np.asarray(bytearray(data), dtype="uint8")
		image = cv2.imdecode(image, cv2.IMREAD_COLOR)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		faces = faceCascade.detectMultiScale(
			gray,
			scaleFactor=1.1,
			minNeighbors=5,
			minSize=(30, 30),
			flags = cv2.cv.CV_HAAR_SCALE_IMAGE
		)
		print "Found {0} faces!".format(len(faces))
		if(len(faces)>0):
			frames = np.empty((len(faces), 32,32, 3))
			frames2 = np.empty((len(faces), 32,32, 3))
			k=0
			for (x,y,w,h) in faces:
				
				frames[k,:,:,:] = cv2.resize(image[y:y+h,x:x+w],(32,32))
				k=k+1
			print frames.shape
			frames2=frames[:,:,:,[2,1,0]]
			k=0
			records= _predictor_(frames2)
			print "minhan is:"
			print  records
			for (x,y,w,h) in faces:
				if(records[k] ==0):
					cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0),3)
				elif(records[k] ==1):
					cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255),3)
				k=k+1
			count=len(glob.glob(settings.BASE_DIR+"/whimh2/static/whimh2/*.jpg"))
			cv2.imwrite(settings.BASE_DIR+"/whimh2/static/whimh2/"+str(count)+".jpg", image)
	return count

def _predictor_(x_train):
	x_train_temp = np.empty((x_train.shape[0],x_train.shape[3],x_train.shape[1],x_train.shape[2]))
	print "there are "+ str(x_train.shape[0]) + " face to predict"
	for j in range(x_train.shape[0]):
		for i in range(x_train.shape[3]):
			x_train_temp[j,i,:,:]=x_train[j,:,:,i]
	x_train=x_train_temp
	test_result=model.loss(x_train)
	return np.argmax(test_result, axis=1)
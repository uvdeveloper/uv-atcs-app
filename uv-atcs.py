import requests
import os
import shutil
from os import listdir
from os.path import isfile, join

#Path to prediction text files generated from detector
DETPATH  = "texts/"
#Path to ground truth json files
ANNOPATH = "json_gt/"
#Path to text file having list of images used for evaluation
IMAGESETFILE = "image_set_file.txt"
#Path to text file having names of the classes
NAMES = "names.txt"
#URL to cloud machine which will return the prediction text files and output images
url_text = "http://40.71.199.201:8081/file_upload"

#imagefiles is a list of all images in images/ folder 
def predict():
	if os.path.isdir("images/"):
		imagefiles = [f for f in listdir(os.getcwd() + "/images/") if isfile(join(os.getcwd() + "/images/", f))]
		
		print("Doing evaluation for "+str(len(imagefiles))+" images")

		for img in imagefiles:
			print("\n")
			print("Processing image ",img)
			files = {'file': open(os.getcwd() + "/images/" + img, 'rb')}
			response_text = requests.request("POST", url_text, files=files)
			filename_w_ext = os.path.basename(os.getcwd() + "/images/" + img)
			filename, file_extension = os.path.splitext(filename_w_ext)

			#Writing predictions to a text file as "Class_ID Confidence X_Min Y_Min X_Max Y_Max" on each line
			file = open("texts/" + filename + ".txt", 'w')
			file.write(response_text.content)
			file.close();
			print("Output predictions text file is written to " ,"texts/"+filename+".txt")

			url_image = "http://40.71.199.201:8081/" + filename_w_ext;
			response_image = requests.get(url_image, stream=True)

			#Writing output images to the output_images folder
			with open("output_images/" + filename + ".jpg", 'wb') as out_file:
				print("Output image is written to ","output_images/" + filename + ".jpg")				
				shutil.copyfileobj(response_image.raw, out_file)
				
			del response_image
	else:
		print("The 'images/' does not exist. The input images should be in images folder.")

#Fn to create all the necessary files and folders required for the evaluation
def util():
	if not os.path.isdir("json_gt"):
		os.system("mkdir json_gt")
	if not os.path.exists("image_set_file.txt"):
		os.system("touch image_set_file.txt")

	os.chdir("images/")
	os.system("ls *jpg > ../image_set_file.txt")
	os.chdir("..")
	os.system("cd ..")

	if not os.path.exists("test_images.txt"):
		os.system("touch test_images.txt")
	else:
		fp = os.path.abspath("images/")
		#print("ls "+fp+"/*jpg > test_images.txt")		
		os.system("ls "+fp+"/*jpg > test_images.txt")
	
	os.system("> result.txt")
	
def eval(detpath,annopath,imagesetfile,fclassname):
	os.system("python eval/voc_eval.py -detpath "+detpath+" -annopath "+annopath+" -imagesetfile "+imagesetfile+" -names_file "+fclassname)
	print("Evaluation done successfully")
	#print("python eval/voc_eval.py -detpath "+detpath+" -annopath "+annopath+" -imagesetfile "+imagesetfile+" -names_file "+fclassname)

util()
predict()
eval(DETPATH,ANNOPATH,IMAGESETFILE,NAMES)

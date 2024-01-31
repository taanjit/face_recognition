from numpy import save
from PIL import Image, ImageDraw
# from IPython.display import display
from face_recognition import api
import face_recognition
import numpy as np
import glob

file_path='./user_face_info/'
known_face_encodings=[]
known_face_names=[]

for img in glob.glob(file_path+"*.png"):
    file_name=(img.split('/')[2]).split('.')[0]
    print(file_name)

    # Load a sample picture and learn how to recognize it.
    image_file = face_recognition.api.load_image_file(img)

    known_face_encodings.append(face_recognition.api.face_encodings(image_file)[0])
    known_face_names.append((img.split('/')[2]).split('.')[0])

save('known_face_encodings.npy', known_face_encodings)
save('known_face_names.npy', known_face_names)

print('Learned encoding for', len(known_face_encodings), 'images.')
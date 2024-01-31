from numpy import save
from PIL import Image, ImageDraw
# from IPython.display import display
from face_recognition import api
import face_recognition
import numpy as np

# The program we will be finding faces on the example below
pil_im = Image.open('two_people.jpg')
print(pil_im)
# display(pil_im)

# This is an example of running face recognition on a single image
# and drawing a box around each person that was identified.

# Load a sample picture and learn how to recognize it.
obama_image = face_recognition.api.load_image_file("obama.jpg")
obama_face_encoding = face_recognition.api.face_encodings(obama_image)[0]

# Load a second sample picture and learn how to recognize it.
biden_image = face_recognition.api.load_image_file("biden.jpg")
biden_face_encoding = face_recognition.api.face_encodings(biden_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    obama_face_encoding,
    biden_face_encoding
]


known_face_names = [
    "Barack Obama",
    "Joe Biden"
]

save('known_face_encodings.npy', known_face_encodings)
save('known_face_names.npy', known_face_names)


print('Learned encoding for', len(known_face_encodings), 'images.')
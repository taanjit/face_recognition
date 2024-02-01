from PIL import Image, ImageDraw
# from IPython.display import display
from face_recognition import api
import face_recognition
import numpy as np
from numpy import load
import glob


known_face_encodings = data = load('known_face_encodings.npy')
known_face_names = data = load('known_face_names.npy')

test_file_path='./test_img/'

for img in glob.glob(test_file_path+"*.jpeg"):
 



    # # Load an image with an unknown face
    unknown_image = face_recognition.api.load_image_file(img)

    # Find all the faces and face encodings in the unknown image
    face_locations = face_recognition.api.face_locations(unknown_image)
    face_encodings = face_recognition.api.face_encodings(unknown_image, face_locations)

    # Convert the image to a PIL-format image so that we can draw on top of it with the Pillow library
    # See http://pillow.readthedocs.io/ for more about PIL/Pillow
    pil_image = Image.fromarray(unknown_image)
    # Create a Pillow ImageDraw Draw instance to draw with
    draw = ImageDraw.Draw(pil_image)

    # Loop through each face found in the unknown image
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for the known face(s)
        matches = face_recognition.api.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

        # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.api.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        print(name)
        # Draw a box around the face using the Pillow module
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        # Draw a label with a name below the face
        text_width, text_height = draw.textlength(str(name)), 12
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))



        pil_image.show()
        # pil_image.save(f'./{out_folder}/{img_name}_bb.jpg')
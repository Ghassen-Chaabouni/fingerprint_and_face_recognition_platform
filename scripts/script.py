
import cv2

cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

img_counter = 0

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "CIN.jpg"
        cv2.imwrite(img_name, frame)
        
        break

cam.release()

cv2.destroyAllWindows()

import cv2
import face_recognition
import extract_face

from extract_face import extract_face 

extract_face(image_path = "CIN.jpg")

try:
    cin_face = face_recognition.load_image_file('CIN_0_.jpg')             # Load the image as numpy array
    cin_face_encoding = face_recognition.face_encodings(cin_face)[0]   # Get the facial features to compare them


    # Create an array of encodings
    known_face_encodings = [
        cin_face_encoding
    ]

    # Create an array of names
    known_face_names = [
        "CIN"
    ]


    cam = cv2.VideoCapture(0)

    cv2.namedWindow("test")

    img_counter = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("test", frame)

        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            img_name = "opencv_frame_{}.png".format(img_counter)

            face_locations = face_recognition.face_locations(frame)
            face_encodings = face_recognition.face_encodings(frame, face_locations)
            try:
                matches = face_recognition.compare_faces(known_face_encodings[0], face_encodings)
            except:
                matches = False
            break

    cam.release()

    cv2.destroyAllWindows()

    print(matches)
except:
    print('No face detected')

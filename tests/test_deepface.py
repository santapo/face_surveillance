import cv2

from face_recognizer import DeepFaceRecognizer

image1 = cv2.imread("/home/ailab/.santapo/face_surveillance/face_db/po/1.jpg")
image2 = cv2.imread("/home/ailab/.santapo/face_surveillance/face_db/guy_1/Alastair_Campbell_0002.jpg")


recognizer = DeepFaceRecognizer()


resp = recognizer.find([image1, image2])
name = recognizer.get_identity_names(resp)

print(name)
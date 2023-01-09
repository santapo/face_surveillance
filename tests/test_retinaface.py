from face_detector.retinaface import RetinaFace
import cv2



image = cv2.imread("tmp/test.jpeg")

detector = RetinaFace()

res = detector.detect_single_image(image)
print(len(res))
extracted_face = detector.extract_face(image, res)

print(len(extracted_face))
for idx, face in enumerate(extracted_face):
    cv2.imwrite(f"face_{idx}.jpg", face)
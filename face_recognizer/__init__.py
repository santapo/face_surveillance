from .deepface import DeepFaceRecognizer

FACE_RECOGNIZER = {
    "deepface": DeepFaceRecognizer
}


def get_face_recognizer(recognizer_name: str = "deepface"):
    face_recognizer = FACE_RECOGNIZER.get(recognizer_name)
    return face_recognizer
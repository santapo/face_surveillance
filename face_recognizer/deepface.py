import logging
import os
import pickle
import time
from typing import List

import cv2
import numpy as np
import pandas as pd
from deepface.basemodels import (ArcFace, DeepID, DlibWrapper, Facenet,
                                 Facenet512, FbDeepFace, OpenFace,
                                 SFaceWrapper, VGGFace)
from deepface.commons import distance as dst
from deepface.commons import functions
from deepface.extendedmodels import Age, Emotion, Gender, Race
from keras.preprocessing import image as keras_image
from tqdm import tqdm

logger = logging.getLogger()

MODELS = {
    'VGG-Face': VGGFace.loadModel,
    'OpenFace': OpenFace.loadModel,
    'Facenet': Facenet.loadModel,
    'Facenet512': Facenet512.loadModel,
    'DeepFace': FbDeepFace.loadModel,
    'DeepID': DeepID.loadModel,
    'Dlib': DlibWrapper.loadModel,
    'ArcFace': ArcFace.loadModel,
    'SFace': SFaceWrapper.load_model,
    'Emotion': Emotion.loadModel,
    'Age': Age.loadModel,
    'Gender': Gender.loadModel,
    'Race': Race.loadModel
}


class DeepFaceRecognizer:
    def __init__(
        self,
        model_name: str = "ArcFace",
        face_db_path: str = "./face_db/",
    ):
        self.model_name = model_name
        if os.path.isdir(face_db_path) == True:
            self.face_db_path = face_db_path
        else:
            raise ValueError("Passed db_path does not exist!")
        self._build_model()
        self._build_face_representations()

    def _build_model(self):
        model = MODELS.get(self.model_name)
        if model:
            self.model = model()
        else:
            raise ValueError('Invalid model_name passed - {}'.format(self.model_name))

    def _build_face_representations(self):
        file_name = "representations_%s.pkl" % (self.model_name)
        file_name = file_name.replace("-", "_").lower()

        if os.path.exists(self.face_db_path+"/"+file_name):
            logger.warn(f"""Representations for images in {self.face_db_path} folder were previously stored in {file_name}.
                        If you added new instances after this file creation, then please delete this file and call find function again.
                        It will create it again.""")
            f = open(self.face_db_path+'/'+file_name, 'rb')
            self.representations = pickle.load(f)
            logger.info("There are ", len(self.representations)," representations found in ",file_name)

        else: #create representation.pkl from scratch
            employees = []
            for r, d, f in os.walk(self.face_db_path): # r=root, d=directories, f = files
                for file in f:
                    if ('.jpg' in file.lower()) or ('.png' in file.lower()):
                        exact_path = r + "/" + file
                        employees.append(exact_path)
            if len(employees) == 0:
                raise ValueError("There is no image in ", self.face_db_path," folder! Validate .jpg or .png files exist in this path.")

            self.representations = []
            pbar = tqdm(range(0,len(employees)), desc='Finding representations')
            for index in pbar:
                employee = employees[index]
                instance = []
                instance.append(employee)
                image = cv2.imread(employee)
                representation = self.represent(image)
                instance.append(representation)
                self.representations.append(instance)
            f = open(self.face_db_path+'/'+file_name, "wb")
            pickle.dump(self.representations, f)
            f.close()
            logger.info("Representations stored in ",self.face_db_path,"/",file_name," file. \
                        Please delete this file when you add new identities in your database.")

    @staticmethod
    def preprocess_face(face_image: np.ndarray, target_size: List[int]) -> np.ndarray:
        if face_image.shape[0] > 0 and face_image.shape[1] > 0:
            factor_0 = target_size[0] / face_image.shape[0]
            factor_1 = target_size[1] / face_image.shape[1]
            factor = min(factor_0, factor_1)

            dsize = (int(face_image.shape[1] * factor), int(face_image.shape[0] * factor))
            face_image = cv2.resize(face_image, dsize)

            # Then pad the other side to the target size by adding black pixels
            diff_0 = target_size[0] - face_image.shape[0]
            diff_1 = target_size[1] - face_image.shape[1]

            face_image = np.pad(face_image, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2), (0, 0)), 'constant')

        # double check whether the face_image has target_size
        if face_image.shape[0:2] != target_size:
            face_image = cv2.resize(face_image, target_size)

        image_pixels = keras_image.img_to_array(face_image)
        image_pixels = np.expand_dims(image_pixels, axis = 0)
        image_pixels /= 255 #normalize input in [0, 1]
        return image_pixels

    @staticmethod
    def get_identity_names(resp_obj):
        identity_names = []
        for obj in resp_obj:
            identity = obj["identity"].tolist()
            identity = [item.split("/")[-2] for item in identity]
            identity = max(identity, key=identity.count)
            identity_names.append(identity)
        return identity_names

    def represent(self, face_image: np.ndarray) -> np.ndarray:
        input_shape_x, input_shape_y = functions.find_input_shape(self.model)
        #detect and align
        face_image = __class__.preprocess_face(face_image, target_size=(input_shape_y, input_shape_x))
        face_image = functions.normalize_input(img=face_image, normalization='base')
        #represent
        embedding = self.model.predict(face_image)[0].tolist()
        return embedding

    def find(
        self,
        face_images: List[np.ndarray],
        prog_bar: bool = True,
    ) -> List[np.ndarray]:

        tic = time.time()
        df = pd.DataFrame(self.representations, columns = ["identity", "%s_representation" % (self.model_name)])
        df_base = df.copy() #df will be filtered in each img. we will restore it for the next item.

        resp_obj = []
        global_pbar = tqdm(range(0, len(face_images)), desc='Analyzing', disable=prog_bar)
        for j in global_pbar:
            face = face_images[j]
            #find representation for passed image
            target_representation = self.represent(face)

            distances = []
            for index, instance in df.iterrows():
                source_representation = instance["%s_representation" % (self.model_name)]
                distance = dst.findCosineDistance(source_representation, target_representation)
                distances.append(distance)

            df["%s_%s" % (self.model_name, "cosine")] = distances
            threshold = dst.findThreshold(self.model_name, "cosine")
            df = df.drop(columns = ["%s_representation" % (self.model_name)])
            df = df[df["%s_%s" % (self.model_name, "cosine")] <= threshold]
            df = df.sort_values(by = ["%s_%s" % (self.model_name, "cosine")], ascending=True).reset_index(drop=True)
            resp_obj.append(df)
            df = df_base.copy() #restore df for the next iteration

        toc = time.time()
        logger.info("find function lasts ",toc-tic," seconds")

        if len(resp_obj) == 1:
            return resp_obj[0]
        return resp_obj
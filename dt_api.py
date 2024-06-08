import PIL.Image
from fastapi import FastAPI, File, UploadFile

# STEP 1: Import the necessary modules.
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# STEP 2: Create an ObjectDetector object.
base_options = python.BaseOptions(model_asset_path='models\\efficientdet_lite0.tflite')
options = vision.ObjectDetectorOptions(base_options=base_options,score_threshold=0.5)
detector = vision.ObjectDetector.create_from_options(options)


app = FastAPI()

import io
import PIL
import numpy as np

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    byte_file = await file.read()

    # STEP 3: Load the input image.
    #image = mp.Image.create_from_file(IMAGE_FILENAMES[1])

    # convert char array to binary array
    image_bin = io.BytesIO(byte_file)
    
    # create PIL Image from binary array
    pil_img = PIL.Image.open(image_bin)

    # Convert MP Image from PIL IMAGE
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(pil_img))

    # STEP 4: Detect objects in the input image.
    detection_result = detector.detect(image)
    # print(detection_result)


    # STEP 5: Process the classification result. In this case, visualize it.
    # count = 3
    # results = []
    # for i in range(count):
    #     category = classification_result.classifications[0].categories[i]
    #     results.append({"category":category.category_name, "score": category.score})
    # # result = f"{top_category.category_name} - ({top_category.score:.2f})"

    # DetectionResult(
    # detections=[
    #     Detection(
    #         bounding_box=BoundingBox(origin_x=72, origin_y=162, width=252, height=191), 
    #         categories=[Category(index=None, score=0.7798683643341064, display_name=None, category_name='cat')], 
    #         keypoints=[]), 
    #     Detection(
    #         bounding_box=BoundingBox(origin_x=303, origin_y=27, width=248, height=344), 
    #         categories=[Category(index=None, score=0.7624295949935913, display_name=None, category_name='dog')], 
    #         keypoints=[])
    #     ]
    # )
    det_result = []
    for detection in detection_result.detections:
        print(detection.categories[0].category_name)
        det_result.append(detection.categories[0].category_name)

    return {"result": det_result}



from fastapi import FastAPI, File, UploadFile
# STEP1.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.components import processors
from mediapipe.tasks.python import vision


# STEP 2: Create an ImageClassifier object. //추론기 먼저 선언하고 fastapi 선언하기
base_options = python.BaseOptions(model_asset_path='models\\efficientnet_lite0.tflite')
options = vision.ImageClassifierOptions(base_options=base_options, max_results=3)
classifier = vision.ImageClassifier.create_from_options(options)

app = FastAPI()


# @app.post("/files/")
# async def create_file(file: bytes = File()):
#     return {"file_size": len(file)}

import io
import PIL
import numpy as np

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):

    byte_file = await file.read()

    # STEP 3: Load the input image.
    # image = mp.Image.create_from_file(IMAGE_FILENAMES[0])//아래 세 단계가 create_from에 포함

    # convert char array to binary array
    image_bin = io.BytesIO(byte_file)  # 바이트형식으로바꿔주고
    
    # create PIL image from binary array
    pil_img = PIL.Image.open(image_bin)    # 이미지로 다시 바꿔줌
   
    # convert MP Image from PIL Image
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(pil_img))


    # STEP 4: Classify the input image.
    classification_result = classifier.classify(image)
    print(classification_result)

    # # STEP 5: Process the classification result. In this case, visualize it.
    # top_category = classification_result.classifications[0].categories[0]
    # # result = f"{top_category.category_name} ({top_category.score:.2f})"

    
    # return {"result": {
    #     "category":top_category.category_name,
    #     "score":top_category.score
    # }}
            
    count = 3
    results = []
    for i in range(count):
        # STEP 5: Process the classification result. In this case, visualize it.
        category = classification_result.classifications[0].categories[i]
        results.append({"category":category.category_name,"score":category.top_category.score })
    # result = f"{top_category.category_name} ({top_category.score:.2f})"
    
    return {"result":results}

############외우기!!!!!!!!!
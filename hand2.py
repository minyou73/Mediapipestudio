# from mediapipe import solutions
# from mediapipe.framework.formats import landmark_pb2
# import numpy as np
# import cv2


# MARGIN = 10  # pixels
# FONT_SIZE = 1
# FONT_THICKNESS = 1
# HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

# def draw_landmarks_on_image(rgb_image, detection_result):
#   hand_landmarks_list = detection_result.hand_landmarks
#   handedness_list = detection_result.handedness
#   annotated_image = np.copy(rgb_image)

#   # Loop through the detected hands to visualize.
#   for idx in range(len(hand_landmarks_list)):
#     hand_landmarks = hand_landmarks_list[idx]
#     handedness = handedness_list[idx]

#     # Draw the hand landmarks.
#     hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
#     hand_landmarks_proto.landmark.extend([
#       landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
#     ])
#     solutions.drawing_utils.draw_landmarks(
#       annotated_image,
#       hand_landmarks_proto,
#       solutions.hands.HAND_CONNECTIONS,
#       solutions.drawing_styles.get_default_hand_landmarks_style(),
#       solutions.drawing_styles.get_default_hand_connections_style())

#     # Get the top left corner of the detected hand's bounding box.
#     height, width, _ = annotated_image.shape
#     x_coordinates = [landmark.x for landmark in hand_landmarks]
#     y_coordinates = [landmark.y for landmark in hand_landmarks]
#     text_x = int(min(x_coordinates) * width)
#     text_y = int(min(y_coordinates) * height) - MARGIN

#     # Draw handedness (left or right hand) on the image.
#     cv2.putText(annotated_image, f"{handedness[0].category_name}",
#                 (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
#                 FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

#   return annotated_image

# # import cv2
# # cap = cv2.VideoCapture(0)
# # while cap.isOpened():
# #     ret, frame = cap.read()
# #     cv2.imshow('Mediapipe Feed', frame)
    
# #     if cv2.waitKey(10) & 0xFF == ord('q'):
# #         break
        
# # cap.release()
# # cv2.destroyAllWindows()


# # STEP 1: Import the necessary modules.
# import mediapipe as mp
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision
# mp_hand_detection = mp.solutions.hand_detection


# # STEP 2: Create an HandLandmarker object.
# base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
# options = vision.HandLandmarkerOptions(base_options=base_options,
#                                        num_hands=2)
# detector = vision.HandLandmarker.create_from_options(options)

# # STEP 3: Load the input image.
# image = mp.Image.create_from_file("image.jpg")

# # STEP 4: Detect hand landmarks from the input image.
# detection_result = detector.detect(image)

# # STEP 5: Process the classification result. In this case, visualize it.
# annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
# # cv2_imshow(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
# cv2.imshow('mp_hand_detection', annotated_image)
# cv2.waitKey(0)




import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# # 이미지 파일의 경우을 사용하세요.:
# IMAGE_FILES = []
# with mp_hands.Hands(
#     static_image_mode=True,
#     max_num_hands=2,
#     min_detection_confidence=0.5) as hands:
#   for idx, file in enumerate(IMAGE_FILES):
#     # 이미지를 읽어 들이고, 보기 편하게 이미지를 좌우 반전합니다.
#     image = cv2.flip(cv2.imread(file), 1)
#     # 작업 전에 BGR 이미지를 RGB로 변환합니다.
#     results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

#     # 손으로 프린트하고 이미지에 손 랜드마크를 그립니다.
#     print('Handedness:', results.multi_handedness)
#     if not results.multi_hand_landmarks:
#       continue
#     image_height, image_width, _ = image.shape
#     annotated_image = image.copy()
#     for hand_landmarks in results.multi_hand_landmarks:
#       print('hand_landmarks:', hand_landmarks)
#       print(
#           f'Index finger tip coordinates: (',
#           f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
#           f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
#       )
#       mp_drawing.draw_landmarks(
#           annotated_image,
#           hand_landmarks,
#           mp_hands.HAND_CONNECTIONS,
#           mp_drawing_styles.get_default_hand_landmarks_style(),
#           mp_drawing_styles.get_default_hand_connections_style())
#     cv2.imwrite(
#         '/tmp/annotated_image' + str(idx) + '.png', cv2.flip(annotated_image, 1))

# 웹캠, 영상 파일의 경우 이것을 사용하세요.:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("카메라를 찾을 수 없습니다.")
      # 동영상을 불러올 경우는 'continue' 대신 'break'를 사용합니다.
      continue

    # 필요에 따라 성능 향상을 위해 이미지 작성을 불가능함으로 기본 설정합니다.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # 이미지에 손 주석을 그립니다.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
    #보기 편하게 이미지를 좌우 반전합니다.
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
 
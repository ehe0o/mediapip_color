import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import matplotlib.pyplot as plt
import cv2

# 모델 경로 지정
model_path_1 = '../models/selfie_multiclass_256x256.tflite'
model_path_2 = '../models/face_landmarker.task'

BaseOptions = mp.tasks.BaseOptions
ImageSegmenter = mp.tasks.vision.ImageSegmenter
ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# ImageSegmenter 옵션 정의
is_options = ImageSegmenterOptions(
    base_options=BaseOptions(model_asset_path=model_path_1),
    running_mode=VisionRunningMode.IMAGE, # 작업의 실행 모드 설정 (IMAGE, VIDEO, LIVE_STREAM)
    output_category_mask=True) #Tru로 설정될 경우 분할 마스크가 uint8로 포함됨.

# face_landmarker 옵션 정의
fl_options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path_2),
    running_mode=VisionRunningMode.IMAGE)

# 색상 정의 (BGR 형식)
colors = [
    [255, 0, 0],  # background: blue
    [255, 255, 0],  # hair: skyblue
    [0, 255, 0],  # body-skin: green
    [255, 0, 128],  # face-skin: purple
    [255, 0, 255],  # clothes: magenta
    [0, 128, 255]  # others: orange
]

# 카테고리 정의
category = ["background", "hair", "body-skin", "face-skin", "clothes", "others"]


def draw_landmarks_on_image(rgb_image, detection_result):
  face_landmarks_list = detection_result.face_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected faces to visualize.
  for idx in range(len(face_landmarks_list)):
    face_landmarks = face_landmarks_list[idx]

    # Draw the face landmarks.
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
    ])
    # 눈썹 윤곽선
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_LEFT_EYEBROW,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_contours_style())


    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_RIGHT_EYEBROW,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_contours_style())

    # 눈동자 랜드마크
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_IRISES,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp.solutions.drawing_styles
          .get_default_face_mesh_iris_connections_style())

    print(mp.solutions.face_mesh.FACEMESH_IRISES)

    # 입술 랜드마크
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_LIPS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_contours_style())

  return annotated_image

def plot_face_blendshapes_bar_graph(face_blendshapes):

  face_blendshapes_names = [face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
  face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]

  face_blendshapes_ranks = range(len(face_blendshapes_names))

  fig, ax = plt.subplots(figsize=(12, 12))
  bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores, label=[str(x) for x in face_blendshapes_ranks])
  ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
  ax.invert_yaxis()


  for score, patch in zip(face_blendshapes_scores, bar.patches):
    plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")

  ax.set_xlabel('Score')
  ax.set_title("Face Blendshapes")
  plt.tight_layout()
  plt.show()


with ImageSegmenter.create_from_options(is_options) as segmenter, FaceLandmarker.create_from_options(fl_options) as landmarker:

    # 이미지 불러오기
    mp_image = mp.Image.create_from_file('../image/sample2.png')
    numpy_image = mp_image.numpy_view()


    # 이미지 분할
    segmented_masks = segmenter.segment(mp_image)
    print(segmented_masks)

    # 얼굴 랜드마크 추출
    face_landmarker_result = landmarker.detect(mp_image)
    print(face_landmarker_result)

    # 결과값에서 category_mask 속성 가져오기
    category_mask = segmented_masks.category_mask
    category_mask_np = category_mask.numpy_view()

    # 컬러 이미지 생성
    h, w = category_mask_np.shape # numpy 배열의 width, height 사이즈 가져옴
    color_image = np.zeros((h, w, 3), dtype=np.uint8) #동일한 크기의 빈 이미지 생성
    for i, color in enumerate(colors): # 각 인덱스(카테고리)에 대한 색상 채워넣기
        color_image[category_mask_np == i] = color

    # 각 카테고리 존재 여부 확인
    for i in range(len(category)):
        is_present = np.isin(category_mask_np, i).any()
        print(f"{category[i]} 존재 여부 : {is_present}")

    bgr_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    annotated_image = draw_landmarks_on_image(bgr_image, face_landmarker_result)

    #결과
    cv2.imshow('Color', color_image)
    cv2.imshow('Annotated Image', annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math

# 모델 경로 지정
model_path_2 = '../models/face_landmarker.task'

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

fl_options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path_2),
    running_mode=VisionRunningMode.IMAGE)

# 홍채 위치에 대한 랜드마크 인덱스
IRIS_LANDMARKS = {
    'left': [468, 469, 470, 471, 472],
    'right': [473, 474, 475, 476, 477]
}
EYEBROW_LANDMARKS = {
    'left':[70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
    'right':[336, 296, 334, 293, 300, 276, 283, 282, 295, 285]
}
LIPS_LANDMARKS = {
    'upper' : [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308],
    'lower' : [375, 321, 405, 314, 17, 84, 181, 91, 146, 61, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78]
}

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

    # 입술 랜드마크
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_LIPS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_contours_style())

  return annotated_image

def create_polygon_mask(image, landmark_indices):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    pts = np.array([(int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])) for landmark in landmark_indices], dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask

def create_mask_from_landmarks(image, landmark_points, indices_lists):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for indices in indices_lists:
        pts = np.array([landmark_points[idx] for idx in indices], dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
    return mask

def calculate_iris_radius(iris_landmarks):
    # 홍채 중심과 주변 랜드마크 간의 거리를 기반으로 평균 반지름을 계산합니다.
    center = iris_landmarks[0]
    radii = [math.hypot(center.x - point.x, center.y - point.y) for point in iris_landmarks[1:]]
    return np.mean(radii)

def calculate_iris_radius(iris_landmarks, image_shape):
    # 홍채 중심과 주변 랜드마크 간의 거리를 기반으로 평균 반지름을 계산합니다.
    # 반지름을 실제 픽셀 단위로 변환하기 위해 이미지의 너비와 높이를 곱합니다.
    image_width, image_height = image_shape[1], image_shape[0]
    center = iris_landmarks[0]
    radii = [
        math.hypot((center.x - point.x) * image_width, (center.y - point.y) * image_height)
        for point in iris_landmarks[1:]
    ]
    return np.mean(radii)

def create_iris_mask(image, iris_landmarks):
    # 이미지에 대한 마스크를 생성합니다.
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    iris_radius = calculate_iris_radius(iris_landmarks, image.shape)
    iris_center = (int(iris_landmarks[0].x * image.shape[1]), int(iris_landmarks[0].y * image.shape[0]))
    # 마스크에 홍채를 나타내는 원을 그립니다.
    cv2.circle(mask, iris_center, int(iris_radius), color=(255), thickness=-1)
    return mask

def extract_average_color(image, mask):
    # 마스크를 적용하여 홍채의 색을 추출합니다.
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    return cv2.mean(masked_image, mask=mask)[:3]

def apply_mask_on_image(image, mask, color=(0, 255, 0)):
    # 마스크에 색을 적용하기 위한 이미지 복사본 생성
    colored_mask = np.zeros_like(image)
    colored_mask[mask == 255] = color

    # 마스크 적용: 원본 이미지와 색이 적용된 마스크를 합침
    masked_image = cv2.addWeighted(image, 1.0, colored_mask, 0.5, 0)
    return masked_image


with FaceLandmarker.create_from_options(fl_options) as landmarker:
    # 이미지 불러오기
    mp_image = mp.Image.create_from_file('../image/sample2.png')
    numpy_image = mp_image.numpy_view()

    # 랜드마크 검출
    face_landmarker_result = landmarker.detect(mp_image)

    # 검출된 랜드마크가 있는지 확인하고 첫 번째 얼굴의 랜드마크 리스트를 가져옵니다.
    if face_landmarker_result.face_landmarks:
        # 첫 번째 얼굴 랜드마크를 face_landmarks 변수에 할당합니다.
        # 여기서 face_landmarks[0]는 첫 번째 랜드마크 리스트를 의미합니다.
        face_landmarks = face_landmarker_result.face_landmarks[0]

        # BGR 이미지로 변환
        bgr_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

        # 홍채 마스크 생성을 위한 랜드마크 포인트 리스트 생성
        left_iris_points = [face_landmarks[i] for i in IRIS_LANDMARKS['left']]
        right_iris_points = [face_landmarks[i] for i in IRIS_LANDMARKS['right']]
        # 랜드마크 포인트 리스트 생성
        left_eyebrow_points = [face_landmarks[i] for i in EYEBROW_LANDMARKS['left']]
        right_eyebrow_points = [face_landmarks[i] for i in EYEBROW_LANDMARKS['right']]
        upper_lip_points = [face_landmarks[i] for i in LIPS_LANDMARKS['upper']]
        lower_lip_points = [face_landmarks[i] for i in LIPS_LANDMARKS['lower']]

        # 홍채 마스크 생성
        left_iris_mask = create_iris_mask(bgr_image, left_iris_points)
        right_iris_mask = create_iris_mask(bgr_image, right_iris_points)
        left_eyebrow_mask = create_polygon_mask(bgr_image, left_eyebrow_points)
        right_eyebrow_mask = create_polygon_mask(bgr_image, right_eyebrow_points)
        upper_lip_mask = create_polygon_mask(bgr_image, upper_lip_points)
        lower_lip_mask = create_polygon_mask(bgr_image, lower_lip_points)

        eyebrows_mask = cv2.bitwise_or(left_eyebrow_mask, right_eyebrow_mask)
        lips_mask = cv2.bitwise_or(upper_lip_mask, lower_lip_mask)

        # 홍채 색 추출
        left_iris_color = extract_average_color(bgr_image, left_iris_mask)
        right_iris_color = extract_average_color(bgr_image, right_iris_mask)
        average_eyebrows_color = extract_average_color(bgr_image, eyebrows_mask)
        average_lips_color = extract_average_color(bgr_image, lips_mask)

        # 추출된 색상 값 출력
        print(f"Left Iris Color: {left_iris_color}")
        print(f"Right Iris Color: {right_iris_color}")
        print(f"Eyebrows Average Color: {average_eyebrows_color}")
        print(f"Lips Average Color: {average_lips_color}")

        annotated_image = draw_landmarks_on_image(bgr_image, face_landmarker_result)

        cv2.imshow('Annotated Image', annotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No face landmarks detected.")
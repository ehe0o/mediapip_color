import mediapipe as mp
import numpy as np
import cv2
import math

# 모델 경로 설정
MODEL_PATH_SEGMENTATION = '../models/selfie_multiclass_256x256.tflite'
MODEL_PATH_LANDMARK = '../models/face_landmarker.task'

# Mediapipe 모델 옵션 설정
BaseOptions = mp.tasks.BaseOptions
ImageSegmenter = mp.tasks.vision.ImageSegmenter
ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


# 랜드마크 인덱스 정의
LANDMARK_INDICES = {
    'iris_left': [468, 469, 470, 471, 472],
    'iris_right': [473, 474, 475, 476, 477],
    'eyebrow_left': [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
    'eyebrow_right': [336, 296, 334, 293, 300, 276, 283, 282, 295, 285],
    'lips_upper': [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308],
    'lips_lower': [375, 321, 405, 314, 17, 84, 181, 91, 146, 61, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78]
}

# Mediapipe 모델 로딩
def load_mediapipe_models():
    is_options = ImageSegmenterOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH_SEGMENTATION),
    running_mode=VisionRunningMode.IMAGE,
    output_category_mask=True)
    fl_options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH_LANDMARK),
    running_mode=VisionRunningMode.IMAGE)
    segmenter = ImageSegmenter.create_from_options(is_options)
    landmarker = FaceLandmarker.create_from_options(fl_options)
    return segmenter, landmarker

# 이미지 로딩
def load_image(image_path):
    return cv2.imread(image_path)

#다각형 마스크 생성 함수
def create_polygon_mask(image, landmark_indices):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    pts = np.array([(int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])) for landmark in landmark_indices], dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask

#홍채 반지름 계산 함수
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

#홍채 마스크 생성 함수
def create_iris_mask(image, iris_landmarks):
    # 이미지에 대한 마스크를 생성합니다.
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    iris_radius = calculate_iris_radius(iris_landmarks, image.shape)
    iris_center = (int(iris_landmarks[0].x * image.shape[1]), int(iris_landmarks[0].y * image.shape[0]))
    # 마스크에 홍채를 나타내는 원을 그립니다.
    cv2.circle(mask, iris_center, int(iris_radius), color=(255), thickness=-1)
    return mask

# 색상 추출 함수
def extract_color(image, mask):
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    return cv2.mean(masked_image, mask=mask)[:3]

# 메인 실행 함수
def main():
    mp_image = mp.Image.create_from_file('../image/sample2.png')
    numpy_image = mp_image.numpy_view()
    segmenter, landmarker = load_mediapipe_models()

    # 이미지 분할 및 랜드마크 검출
    segmented_masks = segmenter.segment(mp_image)
    face_landmarker_result = landmarker.detect(mp_image)

    # 평균 색상 추출 및 출력
    if landmark_result.face_landmarks:
        for feature, indices in LANDMARK_INDICES.items():
            points = [landmark_result.face_landmarks.landmark[i] for i in indices]
            mask = create_mask(image.shape, points)
            color = extract_color(image, mask)
            print(f"{feature.replace('_', ' ').title()} Color: {color}")

if __name__ == "__main__":
    main()

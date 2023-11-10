import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
import cv2

# 모델 경로 지정
model_path = '../models/selfie_multiclass_256x256.tflite'

BaseOptions = mp.tasks.BaseOptions
ImageSegmenter = mp.tasks.vision.ImageSegmenter
ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# ImageSegmenter 옵션 정의
options = ImageSegmenterOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE, # 작업의 실행 모드 설정 (IMAGE, VIDEO, LIVE_STREAM)
    output_category_mask=True) #Tru로 설정될 경우 분할 마스크가 uint8로 포함됨.

# 카테고리 정의
category = ["background", "hair", "body-skin", "face-skin", "clothes", "others"]

def extract_average_color(image, mask):
    masked = cv2.bitwise_and(image, image, mask=mask)
    average_color = cv2.mean(masked, mask=mask)
    return average_color[:3]

with ImageSegmenter.create_from_options(options) as segmenter:

    # 이미지 불러오기
    mp_image = mp.Image.create_from_file('../image/sample2.png')

    # 원본 이미지 가져오기 (알파 채널 제거)
    original_image = mp_image.numpy_view()

    # 이미지 분할
    segmented_masks = segmenter.segment(mp_image)

    # 결과값에서 category_mask 속성 가져오기
    category_mask = segmented_masks.category_mask
    category_mask_np = category_mask.numpy_view()

    hair_mask = np.where(category_mask_np == 1, 255, 0).astype(np.uint8)
    body_skin_mask = np.where(category_mask_np == 2, 255, 0).astype(np.uint8)
    face_skin_mask = np.where(category_mask_np == 3, 255, 0).astype(np.uint8)
    skin_sum_mask = cv2.bitwise_or(body_skin_mask, face_skin_mask)
    clothes_mask = np.where(category_mask_np == 4, 255, 0).astype(np.uint8)
    average_hair_color = extract_average_color(original_image, hair_mask)
    average_skin_color_sum = extract_average_color(original_image, skin_sum_mask)
    average_clothes_color = extract_average_color(original_image, clothes_mask)

    print(f"Hair: {average_hair_color}")
    print(f"Skin_sum:{average_skin_color_sum}")
    print(f"clothes: {average_clothes_color}")



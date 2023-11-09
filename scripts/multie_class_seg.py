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


with ImageSegmenter.create_from_options(options) as segmenter:

    # 이미지 불러오기
    mp_image = mp.Image.create_from_file('../image/sample2.png')

    # 원본 이미지 가져오기 (알파 채널 제거)
    original_image = mp_image.numpy_view()
    original_image = original_image[:, :, :3]  # 알파 채널 제거 -> addWeight를 위한 작업

    # 이미지 분할
    segmented_masks = segmenter.segment(mp_image)
    print(segmented_masks)

    # 결과값에서 category_mask 속성 가져오기
    category_mask = segmented_masks.category_mask
    category_mask_np = category_mask.numpy_view()

    # 각 카테고리 존재 여부 확인
    for i in range(len(category)):
        is_present = np.isin(category_mask_np, i).any()
        print(f"{category[i]} 존재 여부 : {is_present}")



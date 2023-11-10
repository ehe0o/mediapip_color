## Mediapipe_color
- 사용 모델 : mediapipe
  - Face Landmark Detection
  - Image Segmentation(Multi-class selfie segmenter 256)
- 모델 실행 -> 검출된 segmentation 및 랜드마크 마스크 생성 -> 해당 부분의 색상 평균값 구함 (소수점 첫째 자리에서 반올림) <br>
![평균,사진 평균 색상](https://github.com/DAASHeo/mediapip_color/assets/64454313/8ef19bb9-15e1-4154-a2ff-3ead19410d5f)
![image](https://github.com/DAASHeo/mediapip_color/assets/64454313/62dcf3ad-bb44-4706-97cb-440cb72f0531)

- 생각해야할 부분
  1. 클린 코드
  2. 실제 이미지의 색상
  3. 예외 처리 추가

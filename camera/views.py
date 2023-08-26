from django.shortcuts import render
from django.views.decorators import gzip
import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from django.http import StreamingHttpResponse
from camera.models import MyModel
import threading
from PIL import ImageFont, ImageDraw, Image
import torch.nn.functional as F
from django.http import JsonResponse
from collections import deque
from django.core.cache import cache
from django.http import HttpResponse

# 모델 경로와 디바이스 설정
model = MyModel()
device = model.device
model.model.to(device)
model.model.eval()

# 카메라 관련 클래스
class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        (self.grabbed, self.frame) = self.video.read()
        self.frame_buffer = deque(maxlen=16)  # 최근 16프레임을 저장하는 버퍼
        threading.Thread(target=self.update, args=()).start()

    def __del__(self):
        self.video.release()

    def get_frame(self):
        image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)  # OpenCV의 BGR 형식을 RGB로 변환
        _, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()
    
    def read(self):
        grabbed, frame = self.video.read()
        if grabbed:
        # 프레임이 제대로 읽어진 경우에만 BGR 형식으로 변환하여 반환
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return grabbed, frame

    def update(self):
        #while True:
        #    (self.grabbed, self.frame) = self.video.read()  # 프레임을 여기서 업데이트합니다
        #    if not self.grabbed:
        #        break
        frame_buffer = deque(maxlen=16)  # 최근 16프레임을 저장하는 버퍼
        while True:
            (self.grabbed, self.frame) = self.video.read()
            if not self.grabbed:
                break

            frame_buffer.append(self.frame)  # 현재 프레임을 버퍼에 추가

            if len(frame_buffer) == 16:  # 16프레임이 채워졌을 때 분류 수행
                frames_to_process = list(frame_buffer)  # 버퍼의 프레임들을 리스트로 변환
                predicted_class_name, predicted_probability = process_frame(frames_to_process, model)
                print(predicted_class_name)
                print(f"Class Probability: {predicted_probability:.2f}")
    
    def update(self):
        while True:
            self.grabbed, self.frame = self.video.read()
            if not self.grabbed:
                break

            self.frame_buffer.append(self.frame)  # 현재 프레임을 버퍼에 추가


#-*- coding: utf-8 -*- 

# 프레임 처리 및 예측 함수
def process_frame(frame, model):
    # 프레임 전처리
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_tensor = transform(frame_rgb)
    
    num_frames = frame_tensor.shape[0]
    stride = 8  # overlap할 구간 크기
    
    # 리스트를 사용하여 각 구간마다 전처리된 프레임을 저장
    frame_tensors = []
    for i in range(0, num_frames, stride):
        end_idx = min(i + 16, num_frames)  # 현재 구간의 마지막 인덱스
        current_frames = frame_tensor[i:end_idx]  # 현재 구간의 프레임들
        
        # 16프레임이 되지 않는 경우에는 빈 프레임을 추가하여 16프레임으로 만듦
        if current_frames.shape[0] < 16:
            padding = torch.zeros(16 - current_frames.shape[0], *current_frames.shape[1:])
            current_frames = torch.cat((current_frames, padding))
        
        frame_tensors.append(current_frames)
    
    # 리스트로 저장된 전처리된 프레임들을 하나의 텐서로 변환
    frame_tensors = torch.stack(frame_tensors)
    
    # 텐서를 모델 입력 형식에 맞게 변환
    if len(frame_tensors.shape) == 4:  # 4D 텐서인 경우 (batch_size, num_frames, ...)
        frame_tensors = frame_tensors.unsqueeze(0)

    # Add a batch dimension
    frame_tensor = frame_tensor.unsqueeze(0)

    # Move the tensor to the appropriate device
    frame_tensor = frame_tensor.to(device)

    # 모델 예측
    with torch.no_grad():
        # Check the number of channels and expand if necessary
        if frame_tensor.shape[0] == 1:
            frame_tensor = frame_tensor.expand(3, -1, -1, -1)
            frame_tensor = frame_tensor.unsqueeze(0)  # Add batch dimension
        else:
            frame_tensor = frame_tensor.unsqueeze(0)  # Add batch dimension

        output = model.model(frame_tensor)
        probabilities = F.softmax(output, dim=1)
        
    class_mapping = {
        0: "가슴",
        1: "귀",
        2: "너무 아파요",
        3: "머리",
        4: "목",
        5: "무릎",
        6: "발",
        7: "발가락",
        8: "발목",
        9: "배",
        10: "손가락",
        11: "손목",
        12: "어깨",
        13: "팔꿈치",
        14: "허리"
    }
    
    # 예측 결과 가져오기
    _, predicted_idx = torch.max(output.data, 1)
    predicted_class_num = predicted_idx.item()
    predicted_class_name = class_mapping.get(predicted_class_num, "알 수 없음")
    predicted_probability = probabilities[0][predicted_idx].item()
    
    #print(predicted_class_name)
    #print(f"Class Probability: {predicted_probability:.2f}")
    
    return predicted_class_name, predicted_probability

# 비디오 프레임을 가져오고 예측 결과를 반환하는 함수
def get_video_frame(cam, model):
    while True:
        frame = cam.get_frame()
        frame_np = np.frombuffer(frame, np.uint8)  # 바이트 데이터를 NumPy 배열로 변환
        frame_img = cv2.imdecode(frame_np, cv2.IMREAD_COLOR)  # NumPy 배열을 이미지로 디코딩
        frame_img_rgb = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
        predicted_class, predicted_probability = process_frame(frame_img_rgb, model)
        print(predicted_class, predicted_probability)
        
        pil_image = Image.fromarray(frame_img_rgb)

        # 예측 결과 표시
        #font = cv2.FONT_HERSHEY_SIMPLEX
        font = ImageFont.truetype('/Users/song-yeojin/hearoweb/hearo/static/fonts/NanumGothic.ttf',40)
        draw = ImageDraw.Draw(pil_image)
        text = f"Predicted Class: {predicted_class}"
        text_prob = f"Probability: {predicted_probability:.2f}"
        text_position = (10, 30)
        text_position_prob = (10, 80) #확률 위치
        text_color = (0, 255, 0)  # Green color (RGB format)
        draw.text(text_position, text, font=font, fill=text_color) # 클래스
        draw.text(text_position_prob, text_prob, font=font, fill=text_color) #확률
        
        
        frame_img_rgb = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        #cv2.putText(frame_img_rgb, f"Predicted Class: {predicted_class}", (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

        ret, jpeg = cv2.imencode('.jpg', frame_img_rgb)
        frame_bytes = jpeg.tobytes()

        # 프레임 단위로 이미지 반환
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

def initialize_model():
    # 모델을 초기화하고 반환
    model = MyModel()
    return model

# 모델을 캐시에서 가져오거나 초기화
def get_model():
    model = cache.get('your_model_key')
    if model is None:
        model = initialize_model()
        cache.set('your_model_key', model, timeout=None)  # 모델을 캐시에 저장 (timeout=None은 캐시를 영구적으로 유지)
    return model

#웹캠 영상 스트리밍
def camera(request):
    try:
        cam = VideoCamera()  # 웹캠 호출
        model = get_model()
        return StreamingHttpResponse(get_video_frame(cam, model), content_type="multipart/x-mixed-replace;boundary=frame")
    except Exception as e:
        print("에러입니다:", str(e))
        pass
    









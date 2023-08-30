from django.shortcuts import render, redirect
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
from django.http import StreamingHttpResponse

# 모델 경로와 디바이스 설정
model = MyModel()
device = model.device
model.model.to(device)
model.model.eval()

# 스트림 버퍼 클래스
class StreamBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []
        self.lock = threading.Lock()  # Lock 객체 생성

    def add_frame(self, frame):
        with self.lock:
            self.buffer.append(frame)
            if len(self.buffer) > self.buffer_size:
                self.buffer.pop(0)

    def get_clip(self, clip_size, stride):
        with self.lock:
            if len(self.buffer) < clip_size:
                return None
            clip = self.buffer[:clip_size]
            self.buffer = self.buffer[stride:]
            return clip

buffer_size = 8
clip_size = 16
stride = 8
stream_buffer = StreamBuffer(buffer_size)


# # 카메라 관련 클래스 원래코드
# class VideoCamera(object):
#     def __init__(self):
#         self.video = cv2.VideoCapture(0)
#         (self.grabbed, self.frame) = self.video.read() 원래 코드
#         threading.Thread(target=self.update, args=()).start()

#     def __del__(self):
#         self.video.release()

#     def get_frame(self):
#         image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)  # OpenCV의 BGR 형식을 RGB로 변환
#         _, jpeg = cv2.imencode('.jpg', image)
#         return jpeg.tobytes()

#     def update(self):
#         while True:
#             (self.grabbed, self.frame) = self.video.read()  # 프레임을 여기서 업데이트합니다
#             if not self.grabbed:
#                 break


# 카메라 관련 클래스
class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.current_frame = None
        threading.Thread(target=self.update, args=()).start()

    def __del__(self):
        self.video.release()

    def update(self):
        while True:
            ret, frame = self.video.read()
            if not ret:
                break
            self.current_frame = frame
            stream_buffer.add_frame(frame)

    def get_clip(self):
        return stream_buffer.get_clip(clip_size, stride)
    
#-*- coding: utf-8 -*- 
'''
# 프레임 처리 및 예측 함수
def process_frame(frame, model):
    # 프레임 전처리
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_tensor = transform(frame_rgb)
    # frame_tensor = transform(frame)
    if frame_tensor.shape[0] == 1:
        frame_tensor = frame_tensor.expand(3, -1, -1, -1)

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
        # 추가적인 클래스와 번호를 매핑하면 됩니다.
    }


    # 예측 결과 가져오기
    _, predicted_idx = torch.max(output.data, 1)
    predicted_class_num = predicted_idx.item()
    #print(predicted_class)
    predicted_class_name = class_mapping.get(predicted_class_num, "알 수 없음")
    predicted_probability = probabilities[0][predicted_idx].item()
    
    print(predicted_class_name)
    print(f"Class Probability: {predicted_probability:.2f}")
    
    return predicted_class_name, predicted_probability
'''
# 프레임 처리 및 예측 함수
def process_clip(clip, model):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    frames_rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in clip]
    frames_tensor = [transform(frame) for frame in frames_rgb]
    frames_tensor = torch.stack(frames_tensor)

    frames_tensor = frames_tensor.unsqueeze(0)
    frames_tensor = frames_tensor.to(device)

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
        # 추가적인 클래스와 번호를 매핑하면 됩니다.
    }
    _, predicted_idx = torch.max(output.data, 1)
    predicted_class_num = predicted_idx.item()
    #print(predicted_class)
    predicted_class_name = class_mapping.get(predicted_class_num, "알 수 없음")
    predicted_probability = probabilities[0][predicted_idx].item()
    
    print(predicted_class_name)
    print(f"Class Probability: {predicted_probability:.2f}")
    
    return predicted_class_name, predicted_probability

# 비디오 프레임을 가져오고 예측 결과를 반환하는 함수
def get_video_frame(cam, model):
    while True:
        frame = cam.get_frame()
        frame_np = np.frombuffer(frame, np.uint8)  # 바이트 데이터를 NumPy 배열로 변환
        frame_img = cv2.imdecode(frame_np, cv2.IMREAD_COLOR)  # NumPy 배열을 이미지로 디코딩
        frame_img_rgb = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
        predicted_class, predicted_probability = process_clip(frame_img_rgb, model)
        print(predicted_class, predicted_probability)
        
        pil_image = Image.fromarray(frame_img_rgb)

        # 예측 결과 표시
        #font = cv2.FONT_HERSHEY_SIMPLEX
        font = ImageFont.truetype('/Users/dohakim/HearoWeb/hearo/static/fonts/NanumGothic.ttf',40)
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

# 프레임 및 예측 결과를 생성하는 제너레이터
# def gen(camera, model):
#     for frame in get_video_frame(camera, model):
#         # 프레임 단위로 이미지 반환
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
''' 예전 코드
# 웹캠 영상 스트리밍
def camera(request):
    try:
        cam = VideoCamera()  # 웹캠 호출
        return StreamingHttpResponse(get_video_frame(cam, model), content_type="multipart/x-mixed-replace;boundary=frame")
    except Exception as e:
        print("에러입니다:", str(e))
        pass
'''

# 웹캠 영상 스트리밍
def camera(request):
    try:
        cam = VideoCamera()  # 웹캠 호출

        while True:
            clip = cam.get_clip()
            if clip:
                predicted_class = process_clip(clip, model)

                # 클립 처리 로직
                # 예: 어떤 동작이나 액션에 따른 처리를 진행

            frame = cam.current_frame
# 프레임 결과 표시
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, f"Predicted Class: {predicted_class}", (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

            _, jpeg = cv2.imencode('.jpg', frame)
            frame_bytes = jpeg.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

    except Exception as e:
        print("에러입니다:", str(e))
        pass

# new code
# def camera(request):
#     try:
#         cam = VideoCamera()  # 웹캠 호출
#         return render(request, 'camera/camera.html')
#     except Exception as e:
#         print("에러입니다:", str(e))
#         return render(request, 'camera/camera.html')
#
# def redirect_to_mic(request):
#     return redirect('/mic/mic')
#
# def get_camera_stream(cam, model):
#     while True:
#         frame = get_video_frame(cam, model)  # 프레임 가져오기
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')  # 이미지 프레임 반환
#
# def camera_stream(request):
#     cam = VideoCamera()  # 웹캠 호출
#     return StreamingHttpResponse(get_camera_stream(cam, model), content_type="multipart/x-mixed-replace;boundary=frame")
#

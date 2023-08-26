import json
from channels.generic.websocket import AsyncWebsocketConsumer
import asyncio
import torch
from camera.models import MyModel
from camera.views import get_video_frame, process_frame  # 이미지 처리 함수를 가져옵니다

model = MyModel()

import json

# 다른 필요한 모듈 임포트

class YourConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()

    async def disconnect(self, close_code):
        pass

    async def receive(self, text_data):
        data = json.loads(text_data)
        if data.get('action') == 'start_camera':
            try:
                # 카메라에서 프레임 데이터 가져오는 로직을 작성
                frame_data = get_video_frame()  # 예: 카메라에서 프레임 가져오는 함수 호출
                
                # 프레임 데이터를 모델에 전달하고 예측 수행
                predicted_class_name, predicted_probability = process_frame(frame_data, model)

                # 프레임 데이터와 예측 결과를 클라이언트에게 전송
                await self.send(text_data=json.dumps({
                    'action': 'update_frame',
                    'frame_data': frame_data,  # 클라이언트에서 이미지로 표시할 수 있도록 프레임 데이터를 전송
                    'prediction_result': {
                        'class_name': predicted_class_name,
                        'probability': predicted_probability,
                    },
                }))
            except Exception as e:
                # 예외 처리: 카메라 또는 예측에 문제가 있는 경우
                await self.send(text_data=json.dumps({
                    'action': 'error',
                    'error_message': str(e),
                }))

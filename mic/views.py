'''
from django.shortcuts import render, redirect
from .realtimeapicall import RGspeech
import time
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.template import loader
# Create your views here.

def mic(request): #접속 시 첫 화면
    return render(request, 'mic/mic.html')

@csrf_exempt
def apic(request): #api 호출 함수
    gsp = RGspeech()
    stt = ''
    while True:
            # 음성 인식 될때까지 대기 한다.
        new_stt = gsp.getText()
        if new_stt is None:
            break
        stt += new_stt
        print(new_stt)
        print(stt)

        if "끝입니다" in stt:
            # return redirect('/home/camera')
            time.sleep((1))

    time.sleep(0.01)
        # \\break
    # time.sleep(0.01)
    return render(request, 'mic/result.html', {'stt' : stt})
'''
from django.urls import path
from .views import *
from django.contrib import admin
from django.urls import include, path


app_name = "home"

urlpatterns = [
    path('select', select, name="select"), #home->views.py의 select함수를 의미
    # home/select (o) , home/select/ (x)
    path('camera', cam, name="camera"), #웹캠 링크
]
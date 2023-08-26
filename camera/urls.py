from .views import *
from django.urls import path
from . import views
from . import consumers  # consumers 모듈 import

app_name = "cam"

urlpatterns = [
    path('camera', camera, name='camera'),
]




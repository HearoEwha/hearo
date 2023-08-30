'''
from .views import *
from django.urls import path
app_name = "mic"

urlpatterns = [
    path('mic', mic, name="mic"),
    # path('speech-to-text', speech_to_text, name='speech_to_text'),
    path('apic', apic, name='apic'),
]
'''
from channels.routing import ProtocolTypeRouter, URLRouter
from camera import consumers
from django.urls import re_path
from . import consumers

application = ProtocolTypeRouter({
    "websocket": URLRouter([
        re_path(r"ws/camera/camera/$", consumers.YourConsumer.as_asgi()),
    ]),
})
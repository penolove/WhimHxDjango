from django.conf.urls import url

from . import views

app_name = 'whimh2'

urlpatterns = [
    url(r'^$', views.detect, name='detect'),
]
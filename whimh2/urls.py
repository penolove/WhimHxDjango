from django.conf.urls import url

from . import views

app_name = 'whimh2'

urlpatterns = [
    url(r'^$', views.detect,  {'metx': 'cnn'},name='detect'),
    url(r'^tensor$', views.detect, {'metx': 'tensor'},name='tensor'),
    #url(r'^tensor$', views.detect, name='detect',{'metx': 'xx'}),
]
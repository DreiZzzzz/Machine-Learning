from django.urls import path
from . import views # import views module from the same folder

urlpatterns = [
    path('', views.index), # '' => root url
    path('new', views.new)
]
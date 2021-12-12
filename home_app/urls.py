from django.contrib import admin
from django.urls import path
from home_app import views

urlpatterns = [
   path("",views.index,name='home_app'),
   path("about",views.about,name='about'),
   path("contacts",views.contacts,name='contacts'),
   path("classical",views.results,name='classical'),
   path("genres",views.genres,name='genres'),
   path("spotify",views.spotify,name='spotify'),

]
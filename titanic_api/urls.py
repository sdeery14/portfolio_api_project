from django.urls import path, include
from . import views
from rest_framework import routers

router = routers.DefaultRouter()
router.register('titanic_api', views.PassengerViewSet)
urlpatterns = [
    path('api/', include(router.urls)),
    path('titanic_prediction/', views.predictSurvival),
 
] 
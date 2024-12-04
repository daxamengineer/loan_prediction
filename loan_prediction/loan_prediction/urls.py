from django.urls import include, path
from prediction import views

urlpatterns = [
    path('', views.predict, name='predict'),
    path('prediction/', include('prediction.urls')),
]

from django.contrib import admin
from django.urls import path, include
from sneakerapp import urls as app_project_urls

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('sneakerapp.urls'))
]

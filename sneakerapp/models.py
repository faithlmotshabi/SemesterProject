from django.db import models

# Create your models here.
class Features(models.Model):
    review_text = models.TextField()

    class Meta:
        app_label = 'sneakerapp'
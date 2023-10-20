from django.shortcuts import render
from django.http import JsonResponse
from rest_framework.response import Response
from rest_framework.decorators import api_view
from rest_framework import status
import pickle 
from .models import Features
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np


# Create your views here.

@api_view(['POST'])
def prediction(request):
    with open('C:/Users/Student/Desktop/SemesterProject/sneakerscript.pkl', 'rb') as file:
        model, vectorizer = pickle.load(file)
        saved_data = pickle.load(file)
        loaded_vectorizer = saved_data['vectorizer']
        loaded_model = saved_data['model']
    data = request.data['review']
    data = loaded_vectorizer.transform(data)
    prediction = loaded_model.predict([data])
    def final_pred(prediction):
        if rating >= 4:
            return 'positive'
        elif rating == 3:
            return 'neutral'
        else:
            return 'negative'

    return Response({'prediction': prediction[0]})



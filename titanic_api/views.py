import numpy as np
import pandas as pd
from django.shortcuts import render
from rest_framework import viewsets
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.http import JsonResponse
from . serializers import PassengerSerializer
from . models import Passenger
from joblib import load

# Create your views here.

class PassengerViewSet(viewsets.ModelViewSet):
    serializer_class = PassengerSerializer
    queryset = Passenger.objects.all()

@api_view(["POST"])
def predictSurvival(request):
	try:
		X = pd.DataFrame(np.array(list(request.data.values())).reshape(1,-1), 
			columns=['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex', 'Embarked'])
		predictor = load("titanic_predictor.joblib")
		prediction = "Survived: Yes" if predictor.predict(X) == 1 else "Survived: No"
		return JsonResponse(prediction, safe=False)
	except ValueError as e:
		return Response(e.args[0], status.HTTP_400_BAD_REQUEST)
from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator

# Create your models here.

class Passenger(models.Model):
	PCLASS1 = 1
	PCLASS2 = 2
	PCLASS3 = 3
	PCLASS_CHOICES=[(PCLASS1,'1'), (PCLASS2,'2'), (PCLASS3, '3')]
	pclass = models.IntegerField(choices=PCLASS_CHOICES)

	age = models.FloatField(validators=[MinValueValidator(0), MaxValueValidator(100)])
	sibsp = models.IntegerField(validators=[MinValueValidator(0), MaxValueValidator(20)])
	parch = models.IntegerField(validators=[MinValueValidator(0), MaxValueValidator(20)])
	fare = models.FloatField(validators=[MinValueValidator(0), MaxValueValidator(1000)])

	SEX_CHOICES = [('male', 'Male'), ('female', 'Female')]
	sex = models.CharField(choices=SEX_CHOICES, max_length=7)

	EMBARKED_CHOICES = [('C', 'Cherbourg'), ('Q', 'Queenstown'), ('S', 'Southampton')]
	embarked = models.CharField(choices=EMBARKED_CHOICES, max_length=1)
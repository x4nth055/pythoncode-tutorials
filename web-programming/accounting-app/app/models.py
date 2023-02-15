from django.db import models
from django.contrib.auth.models import User

# Create your models here.

class Portfolio(models.Model):
	user = models.ForeignKey(User, on_delete=models.CASCADE, null=True)
	name = models.CharField(max_length=30)
	
	def __str__(self):
		return self.name

class Transaction(models.Model):
	journal_list = models.ForeignKey(Portfolio,on_delete=models.CASCADE)
	trans_name = models.CharField(max_length=30)
	trans_type = models.CharField(max_length=3)
	amount = models.IntegerField()
	date = models.DateField()

	def __str__(self):
		return self.trans_name
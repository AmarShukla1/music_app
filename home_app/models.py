from django.db import models

# Create your models here.
#its like a database
class contact(models.Model):
 name=models.CharField(max_length=200)
 #jaise varchar hota hai database me
 email=models.CharField(max_length=200)
 phone=models.CharField(max_length=12)
 desc=models.TextField(max_length=500)
 date=models.DateField()
 

 def __str__(self):
    return self.name

     




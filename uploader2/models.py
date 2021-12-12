from django.db import models

class Upload2(models.Model):
	upload_file = models.FileField()    
	upload_date = models.DateTimeField(auto_now_add =True)

	
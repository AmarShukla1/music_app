from django.views.generic.edit import CreateView


from .models import Upload

class UploadView(CreateView):
    model = Upload
    fields = ['upload_file', ]
    success_url = '../genres'

    
from django.views.generic.edit import CreateView


from .models import Upload2

class UploadView(CreateView):
    model = Upload2
    fields = ['upload_file', ]
    success_url = '../classical'

    
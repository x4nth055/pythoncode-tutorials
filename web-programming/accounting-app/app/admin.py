from django.contrib import admin
from .models import Portfolio, Transaction

# Register your models here.
admin.site.register(Portfolio)
admin.site.register(Transaction)
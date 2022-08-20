from .models import Book
from django.forms import ModelForm
from django import forms

# declaring the ModelForm
class EditBookForm(ModelForm):
    
    class Meta:
        # the Model from which the form will inherit from
        model = Book
        # the fields we want from the Model
        fields = '__all__'
        # styling the form with bootstrap classes
        widgets = {
             'title': forms.TextInput(attrs={'class': 'form-control'}),
             'author': forms.TextInput(attrs={'class': 'form-control'}),
             'price': forms.TextInput(attrs={'class': 'form-control'}),
             'isbn': forms.TextInput(attrs={'class': 'form-control'}),
             
        }

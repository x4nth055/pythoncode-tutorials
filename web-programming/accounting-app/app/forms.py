from django import forms

class createjournal(forms.Form):
    journal_name = forms.CharField(label='Journal Name',max_length=30)

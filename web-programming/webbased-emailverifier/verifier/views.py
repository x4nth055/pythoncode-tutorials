from django.shortcuts import render
# this displays flash messages or notifications
from django.contrib import messages
# importing validate_email and EmailNotValidError
from email_validator import validate_email, EmailNotValidError


# Create your views here.
def index(request):
    # checking if the method is POST
    if request.method == 'POST':
        # getting the email from the form input
        email = request.POST.get('email-address')
        # this is the context
        context = {
            'email': email
        }
        # the try statement for verify/validating the email
        try:
            # validating the actual email address using the validate_email function
            email_object = validate_email(email)
            # creating the message and storing it
            messages.success(request, f'{email} is a valid email address!!')
            # rendering the results to the index page
            return render(request, 'verifier/index.html', context)
        # the except statement will capture EmailNotValidError error
        except EmailNotValidError as e:
            # creating the message and storing it
            messages.warning(request, f'{e}')
            # rendering the error to the index page
            return render(request, 'verifier/index.html', context)

    # this will render when there is no request POST or after every POST request
    return render(request, 'verifier/index.html')

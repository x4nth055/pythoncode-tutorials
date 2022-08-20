from django.db import models


# the Book model with its fields
class Book(models.Model):
    title = models.CharField(max_length=100)
    author = models.CharField(max_length=100)
    price = models.DecimalField(max_digits=10, decimal_places=2)
    isbn = models.CharField(max_length=100)
    # this is the image for a book, the image will be uploaded to images folder
    image = models.ImageField(null=False, blank=False, upload_to='images/')
    created_at = models.DateTimeField(auto_now_add=True, null=True, blank=True)
    
    # this is the string represantation, what to display after querying a book/books
    def __str__(self):
        return f'{self.title}'
    
    # this will order the books by date created
    class Meta:
        ordering = ['-created_at']


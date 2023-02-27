from django.db import models


class Journalist(models.Model):
    first_name = models.CharField(max_length=60)
    last_name = models.CharField(max_length=60)
    bio = models.CharField(max_length=200)
    def __str__(self):
        return f"{ self.first_name } - { self.last_name }"

class Article(models.Model):
    author = models.ForeignKey(Journalist,
                            on_delete=models.CASCADE,
                            related_name='articles')
    title = models.CharField(max_length=120)
    description = models.CharField(max_length=200)
    body = models.TextField()
    location = models.CharField(max_length=120)
    publication_date = models.DateField(auto_now_add=True)

    def __str__(self):
        return f"{ self.author } - { self.title }"

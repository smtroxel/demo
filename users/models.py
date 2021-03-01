from django.contrib.auth.models import AbstractUser
from django.core.exceptions import ValidationError
from django.db import models
from django.utils.translation import ugettext_lazy as lazy_

from .managers import EmailUserManager


class EmailUser(AbstractUser):
    email = models.EmailField(
        lazy_('email address'),
        unique=True,
        help_text=lazy_('Required. Will be used for verification and password reset.'),
        error_messages={
            'unique': lazy_("A user with that email already exists."),
        }
    )

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = []

    objects = EmailUserManager()

    def save(self, *args, **kwargs):
        """
        Overriding save to prevent case insensitive duplication by
        calling the case insensitive filter method from the Manager.
        """
        if not self.pk:
            if self.__class__.objects.filter(username=self.username).exists():
                raise ValidationError("A user with that username already exists.")
            if self.__class__.objects.filter(email=self.email).exists():
                raise ValidationError("A user with that email already exists.")
        super().save(*args, **kwargs)

    def has_profile(self):
        """
        Determines if user has profile by counting
        if they have at least one SurveyTopicType
        """
        return self.survey_topic_types.count() > 0

    def __repr__(self):
        return "User(email={}, username={}, is_active={}, is_superuser={})".format(
            self.email, self.username, self.is_active, self.is_superuser
        )

    def __str__(self):
        return self.__repr__()

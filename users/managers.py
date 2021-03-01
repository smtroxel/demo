from django.contrib.auth.base_user import BaseUserManager
from django.utils.translation import ugettext_lazy as lazy_


class EmailUserManager(BaseUserManager):
    """
    Custom user model manager where email is the unique identifier
    for authentication instead of username.
    """

    def get(self, **kwargs):
        """
        Override get for username and email case insensitive get.
        """
        if 'username' in kwargs:
            kwargs['username__iexact'] = kwargs['username']
            del kwargs['username']
        elif 'email' in kwargs:
            kwargs['email__iexact'] = kwargs['email']
            del kwargs['email']
        return super().get(**kwargs)

    def filter(self, **kwargs):
        """
        Override filter for username and email case insensitive filter.
        """
        if 'username' in kwargs:
            kwargs['username__iexact'] = kwargs['username']
            del kwargs['username']
        elif 'email' in kwargs:
            kwargs['email__iexact'] = kwargs['email']
            del kwargs['email']
        return super().filter(**kwargs)

    def create_user(self, email, password, username, **extra_fields):
        """
        Create and save a User with the given email and password.
        """
        if not email:
            raise ValueError(lazy_('The Email must be set.'))
        if not username:
            raise ValueError(lazy_('The Username must be set.'))
        email = self.normalize_email(email)
        user = self.model(email=email, username=username, **extra_fields)
        user.set_password(password)
        user.save()
        return user

    def create_superuser(self, email, password, username, **extra_fields):
        """
        Create and save a SuperUser with the given email and password.
        """
        extra_fields.setdefault('is_staff', True)
        extra_fields.setdefault('is_superuser', True)
        extra_fields.setdefault('is_active', True)

        if extra_fields.get('is_staff') is not True:
            raise ValueError(lazy_('Superuser must have is_staff=True.'))
        if extra_fields.get('is_superuser') is not True:
            raise ValueError(lazy_('Superuser must have is_superuser=True.'))
        return self.create_user(email, password, username, **extra_fields)
from django import forms
from django.contrib.auth import authenticate, password_validation
from django.contrib.auth.forms import (
    AuthenticationForm,
    PasswordChangeForm,
    SetPasswordForm,
    UserChangeForm,
    UserCreationForm,
    PasswordResetForm,
)
from django.utils.translation import ugettext_lazy as lazy_

from .models import EmailUser


class EmailUserCreationForm(UserCreationForm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['email'].widget.attrs.update({'autofocus': False})

    password1 = forms.CharField(
        label=lazy_("Password"),
        strip=False,
        widget=forms.PasswordInput(
            attrs={
                'autocomplete': 'new-password',
                'placeholder': 'Password'
            }
        ),
        help_text=password_validation.password_validators_help_text_html(),
    )

    password2 = None

    class Meta(UserCreationForm.Meta):
        model = EmailUser
        fields = ('email', 'username',)
        widgets = {
            'email': forms.EmailInput(
                attrs={
                    'autocomplete': 'email',
                    'autocapitalize': 'none',
                    'placeholder': 'Email'
                }
            ),
            'username': forms.TextInput(
                attrs={
                    'autocomplete': 'username',
                    'autocapitalize': 'none',
                    'placeholder': 'Username'
                }
            ),
        }

    def clean_password1(self):
        password = self.cleaned_data.get('password1')
        if password:
            try:
                password_validation.validate_password(password, self.instance)
            except forms.ValidationError as error:
                self.add_error('password1', error)
        return password


class EmailUserChangeForm(UserChangeForm):
    class Meta(UserChangeForm.Meta):
        model = EmailUser
        fields = ('email', 'username')


class EmailUserAuthenticationForm(AuthenticationForm):

    username = forms.CharField(
        label=lazy_("Email/Username"),
        widget=forms.TextInput(
            attrs={
                'autofocus': True,
                'autocomplete': 'email',
                'autocapitalize': 'none',
                'placeholder': 'Email/Username'
            }
        ),
    )

    password = forms.CharField(
        label=lazy_("Password"),
        strip=False,
        widget=forms.PasswordInput(
            attrs={
                'autocomplete': 'new-password',
                'placeholder': 'Password'
            }
        ),
    )

    error_messages = {
        'invalid_login': lazy_(
            "Please enter a correct %(username)s and password."
        ),
        'inactive': lazy_("This account is inactive."),
    }

    def clean(self):
        username = self.cleaned_data.get('username')
        password = self.cleaned_data.get('password')

        if username is not None and password:
            self.user_cache = authenticate(
                self.request, username=username, password=password)
            if self.user_cache is None:
                try:
                    email_user = EmailUser.objects.get(username=username)
                    self.user_cache = authenticate(
                        self.request, username=email_user.email, password=password)
                except EmailUser.DoesNotExist:
                    raise self.get_invalid_login_error()

            if self.user_cache is None:
                raise self.get_invalid_login_error()
            else:
                self.confirm_login_allowed(self.user_cache)

        return self.cleaned_data

    class Meta:
        model = EmailUser


class EmailUserSetPasswordForm(SetPasswordForm):
    """
    A form that lets an email user change set their password without entering the old
    password, but does not require password confirmation.
    """
    new_password1 = forms.CharField(
        label=lazy_("Password"),
        strip=False,
        widget=forms.PasswordInput(
            attrs={
                'autocomplete': 'new-password',
                'placeholder': 'New Password'
            }
        ),
        help_text=password_validation.password_validators_help_text_html(),
    )

    new_password2 = None

    def clean_password1(self):
        password = self.cleaned_data.get('password1')
        if password:
            password_validation.validate_password(password, self.user)
        return password

    class Meta:
        model = EmailUser


class EmailUserPasswordChangeForm(EmailUserSetPasswordForm):
    """
    A form that lets an email user change their password by entering their old
    password, but does not require password confirmation.
    """

    error_messages = {
        **SetPasswordForm.error_messages,
        'password_incorrect': lazy_("Your old password was entered incorrectly. Please enter it again."),
    }

    old_password = forms.CharField(
        label=lazy_("Old password"),
        strip=False,
        widget=forms.PasswordInput(
            attrs={
                'autocomplete': 'current-password',
                'placeholder': 'Old Password',
                'autofocus': True,
            }
        ),
    )

    field_order = ['old_password', 'new_password1', ]

    def clean_old_password(self):
        """
        Validate that the old_password field is correct.
        """
        old_password = self.cleaned_data["old_password"]
        if not self.user.check_password(old_password):
            raise forms.ValidationError(
                self.error_messages['password_incorrect'],
                code='password_incorrect',
            )
        return old_password


class EmailUserPasswordResetForm(PasswordResetForm):
    email = forms.EmailField(
        label=lazy_("Email"),
        max_length=254,
        widget=forms.EmailInput(
            attrs={
                'autofocus': True,
                'autocomplete': 'email',
                'autocapitalize': 'none',
                'placeholder': 'Email'
            }
        ),
    )

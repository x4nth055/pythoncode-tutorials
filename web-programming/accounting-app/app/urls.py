from django.urls import path
from .views import UserSignup, UserLogin, PortfolioCreate, PortfolioList, Journal, PortfolioDelete, TrialBalance
from django.contrib.auth.views import LogoutView

urlpatterns = [
	path('signup/', UserSignup.as_view(), name='signup'),
	path('login/', UserLogin.as_view(), name='login'),
	path('logout/', LogoutView.as_view(next_page='login'), name='logout'),
	path('pfl-create/', PortfolioCreate.as_view(), name='pfl-create'),
	path('', PortfolioList.as_view(), name='pfl-list'),
	path('pfl-journal/pk=<int:pk>', Journal.as_view(), name='pfl-detail'),
	path('pfl-delete/pk=<int:pk>', PortfolioDelete.as_view(), name='pfl-delete'),
	path('pfl-tb/pk=<int:pk>', TrialBalance.as_view(), name='trial-balance')
]
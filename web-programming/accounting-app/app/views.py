from django.shortcuts import render,redirect

from django.views.generic import View
from django.views.generic.detail import DetailView
from django.views.generic.edit import DeleteView, FormView
from django.urls import reverse_lazy

from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import login
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.auth.views import LoginView

from .models import Portfolio
from json import dumps

# Create your views here.
class UserSignup(FormView):
	template_name = 'app/signup.html'
	form_class = UserCreationForm
	redirect_authenticated_user = True
	success_url = reverse_lazy('pfl-list')


	def form_valid(self, form):
		user = form.save()
		if user is not None:
			login(self.request, user)
		return super(UserSignup, self).form_valid(form)

	def get(self, *args, **kwargs):
		if self.request.user.is_authenticated:
			return redirect('pfl-list')
		return super(UserSignup, self).get(*args, **kwargs)


class UserLogin(LoginView):
	template_name = 'app/signin.html'
	fields = '__all__'
	redirect_authenticated_user = True

	def get_success_url(self):
		return reverse_lazy('pfl-list')


class PortfolioList(LoginRequiredMixin,View):
	def get(self,request):
		account = User.objects.get(username=request.user)
		context = {'portfolio':account}
		return render(request,'app/home.html',context)


class PortfolioCreate(LoginRequiredMixin,View):
	def get(self,request):
		return render(request,'app/portfolio_create_form.html')

	def post(self,request):
		user = User.objects.get(username=request.user)
		pfl_name = request.POST.get('portfolio_name')
		user.portfolio_set.create(name=pfl_name)
		my_object = user.portfolio_set.get(name=pfl_name).id
		return redirect('pfl-detail', my_object)


class Journal(LoginRequiredMixin,DetailView):
	model = Portfolio
	template_name = 'app/journal.html'
	context_object_name = 'pfl'

	def get(self,*args,**kwargs):
		return super(Journal, self).get(*args,**kwargs)

	def post(self,*args,**kwargs):
		return super(Journal, self).get(*args,**kwargs)

	def dispatch(self,request,pk,*args,**kwargs):
		dbt_trans, dbt_amt = request.POST.get('dbt'), request.POST.get('dbt-amt')
		cdt_trans, cdt_amt = request.POST.get('cdt'), request.POST.get('cdt-amt')
		trans_date = request.POST.get('trans-date')
		pfl = self.model.objects.get(id=pk)
		if self.request.POST.get('save'):
			try:
				if dbt_trans and dbt_amt and cdt_trans and cdt_amt != None:
					dbt_whole_trans = pfl.transaction_set.create(trans_name=dbt_trans, trans_type='dbt', amount=dbt_amt, date=trans_date)
					cdt_whole_trans = pfl.transaction_set.create(trans_name=cdt_trans, trans_type='cdt', amount=cdt_amt, date=trans_date)
					dbt_whole_trans.save()
					cdt_whole_trans.save()
					print(True)
			except:
				return super(Journal, self).dispatch(request,*args,**kwargs)
		return super(Journal, self).dispatch(request,*args,**kwargs)


class PortfolioDelete(LoginRequiredMixin,DeleteView):
	model = Portfolio
	success_url = reverse_lazy('pfl-list')


def trial_balance_computer(pk):
	pfl = Portfolio.objects.get(id=pk)
	trans_total = {}
	tb_table = []
	tb_total = [0, 0]
	for trans in pfl.transaction_set.all():
		if trans.trans_name not in trans_total:
			trans_total[trans.trans_name] = 0
		if trans.trans_type == 'dbt':
			trans_total[trans.trans_name] += trans.amount
		else:
			trans_total[trans.trans_name] -= trans.amount
	for x in trans_total:
		if trans_total[x] > 0:
			tb_table.append((x, trans_total[x], ''))
			tb_total[0] += trans_total[x]
		elif trans_total[x] < 0:
			tb_table.append((x, '', trans_total[x]))
			tb_total[1] += trans_total[x]
	tb_table.append(('Total:', tb_total[0], tb_total[1]))
	return pfl.name, tb_table


def t_accounts(pk):
	pfl = Portfolio.objects.get(id=pk)
	ledger = {}
	for trans in pfl.transaction_set.all():
		if trans.trans_name not in ledger:
			ledger[trans.trans_name] = []
		if trans.trans_type == 'dbt':
			ledger[trans.trans_name].append(trans.amount)
		else:
			ledger[trans.trans_name].append(-trans.amount)
	return ledger


class TrialBalance(LoginRequiredMixin, View):
    def get(self, request, pk):
        tb = trial_balance_computer(pk)
        ta = t_accounts(pk)
        ta_JSON = dumps(ta)
        context = {'pk':pk, 'name':tb[0], 'tb':tb[1], 'ta':ta_JSON}
        return render(request, 'app/trialbalance.html', context)

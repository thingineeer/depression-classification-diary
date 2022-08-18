from django.shortcuts import render

# Create your views here.

from django.shortcuts import render,get_object_or_404,redirect
from django.utils import timezone
from .models import Question
from django.http import HttpResponseNotAllowed
from .forms import QuestionForm,AnswerForm
from django.core.paginator import Paginator

from .switch import main


def index(request):
    page = request.GET.get('page','1') #페이지
    question_list = Question.objects.order_by('-create_date')
    paginator = Paginator(question_list,10) # 페이지당 10개씩 보여주기
    page_obj = paginator.get_page(page)
    max_index = len(paginator.page_range)
    context = {'question_list': page_obj,'max_index':max_index}

    return render(request,'pybo/question_list.html',context)

def detail(request,question_id):

    question = get_object_or_404(Question,pk=question_id)
    context ={'question':question}
    return render(request,'pybo/question_detail.html',context)


def answer_create(request,question_id):
    question = get_object_or_404(Question,pk=question_id)

    if request.method =='POST':
        form = AnswerForm(request.POST)
        if form.is_valid():
            answer = form.save(commit=False)
            answer.content = main.main(answer.content)
            answer.create_date = timezone.now()
            answer.question = question
            answer.save()
            return redirect('pybo:detail',question_id=question.id)
    else:
        return HttpResponseNotAllowed('Only POST is possible')

    context = {'question':question,'form':form}
    return render(request,'pybo/question_detail.html',context)

def question_create(request):
    if request.method == 'POST':
        form = QuestionForm(request.POST)
        if form.is_valid():
            question = form.save(commit=False)
            question.create_date = timezone.now()
            question.save()
            return redirect('pybo:index')
    else:
        form = QuestionForm()
    context = {'form': form}
    return render(request, 'pybo/question_form.html', context)

def question_modify(request,question_id):
    question = get_object_or_404(Question,pk=question_id)
    if request.method == 'POST':
        form = QuestionForm(request.POST,instance=question)
        if form.is_valid():
            form.save()
            return redirect('pybo:detail',question_id=question.id)
    else:
        form = QuestionForm(instance=question)
    context = {'form':form}
    return render(request,'pybo/question_form.html',context)

def question_delete(request,question_id):
    question = get_object_or_404(Question,pk=question_id)
    question.delete()
    return redirect('pybo:index')

def answer_modify(request,answer_id):
    answer = get_object_or_404(Question,pk=answer_id)
    if request.method == 'POST':
        form = AnswerForm(request.POST,instance=answer)
        if form.is_valid():
            form.save()
            return redirect('pybo:detail',question_id=answer.question.id)
    else:
        form = AnswerForm(instance=answer)
    context = {'form':form}
    return render(request,'pybo/question_form.html',context)

def answer_delete(request,answer_id):
    answer = get_object_or_404(Question,pk=answer_id)
    answer.delete()
    return redirect('pybo:detail',question_id=answer.question.id)
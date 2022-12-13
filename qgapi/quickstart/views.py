import json
from django.http import HttpResponse
# Create your views here.

from django.contrib.auth.models import User, Group
from rest_framework import viewsets
from rest_framework import permissions

from transformers import AutoTokenizer,T5ForConditionalGeneration
import re
from django.views.decorators.csrf import csrf_exempt

model = T5ForConditionalGeneration.from_pretrained("./models")
tokenizer = AutoTokenizer.from_pretrained('t5-small')

@csrf_exempt
def questionGeneration(request):
    if request.method =='GET':
        return HttpResponse("Reponse for GET request from /question-generation")

    if request.method =='POST':
        print(request)
        data = json.loads(request.body)
        sentence = data['sentence']
        sentence=tokenizer(sentence,return_tensors="pt")
        outs = model.generate(input_ids=sentence['input_ids'], attention_mask=sentence['attention_mask'],max_length=512,early_stopping=True,num_beams=10,num_return_sequences=10)
        outs=[tokenizer.decode(ids) for ids in outs]
        questions=[]
        for s in outs:
            s=re.sub(r'<pad>', '', s)
            s=re.sub(r'</s>', '', s)
            questions.append(s)
        print(questions)
        return HttpResponse([questions])
        
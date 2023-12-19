# ra-bot-c
## Ask any Question from Any Specific Website
This Website Framework allows users to to ask any question about any Website by user choice. This model is made with Python Django Framwork with Langchain. It uses Mutiple Text Embedding (All MiniLM L6 V2, Intfloat E5 Base, Intfloat E5 Large V2) and Mutiple LLMs like Open AI, Google T5 Flan, Flan Alpaca GPT4, Llama 2.User can use custom created Vectorstores and also Visualize them. 

**LLMs like Flan Alpaca GPT4 and Llama 2 are downloaded loacally and uses local RAM and GPU.**


#### To Run Website:

- Add .env file with HUGGINGFACEHUB API TOKEN to Load Open Source LLM (Google T5 Flan, Flan Alpaca GPT4 and Llama 2) and OPENAI API KEY for Open AI LLM
- To Start Website :
```
 python manage.py runserver
```

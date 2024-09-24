import gradio as gr
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

model_name = "IProject-10/roberta-base-finetuned-squad2"
nlp = pipeline("question-answering", model=model_name, tokenizer=model_name)

def predict(context, question):
    res = nlp({"question": question, "context": context})
    return res["answer"]

md = """
### Description

In this project work we build a **Text Extraction Question-Answering system** using **BERT** model. QA system is a important NLP task in which the user asks a question in natural language to the model as input and the model provides the answer in natural language as output.
The language representation model BERT stands for **Bidirectional Encoder Representations from Transformers**. The model is based on the Devlin et al. paper: [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805).
Dataset used is **SQuAD 2.0** [Stanford Question Answering Dataset 2.0](https://rajpurkar.github.io/SQuAD-explorer/). It is a reading comprehension dataset which consists of question-answer pairs derived from wikipedia articles written by crowdworkers.
The answer to all the questions is in the form of a span of text.


### Design of the system:
<br>
<div style="text-align: center;">
    <img src="https://i.imgur.com/G4qgMhE.jpeg" alt="Description Image" style="border: 2px solid #000; border-radius: 5px; width: 600px; height: auto; display: block; margin: 0 auto;">
</div>

### QA Application:
Add a context paragraphs upto 512 tokens and ask a question based on the context. The model acccurately fetches the answer from the context in the form of a text span and display it.

"""

context = "The Amazon rainforest, also known in English as Amazonia or the Amazon Jungle, is a moist broadleaf forest that covers most of the Amazon basin of South America..."
question = "Which continent is the Amazon rainforest in?"

gr.Interface(
    predict,
    inputs=[
        gr.Textbox(lines=7, value=context, label="Context Paragraph"),
        gr.Textbox(lines=2, value=question, label="Question"),
    ],
    outputs=gr.Textbox(label="Answer"),
    title="Question & Answering with BERT using the SQuAD 2 dataset",
    description=md,
).launch()

import gradio as gr
import requests
import json

# Mistral API 키를 여기에 입력하세요.
MISTRAL_API_KEY = 'iIjRdeaRwl9lx7k7A68SsZqCV2NJzAdv'

def chat_with_mistral(message, history):
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Authorization': f'Bearer {MISTRAL_API_KEY}'
    }

    data = {
        'model': 'mistral-tiny',
        'messages': [{'role': 'user', 'content': message}]
    }

    response = requests.post('https://api.mistral.ai/v1/chat/completions', headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        response_json = response.json()
        return response_json['choices'][0]['message']['content']
    else:
        return "Error: Unable to get response from Mistral API"

chat_interface = gr.ChatInterface(fn=chat_with_mistral, title="Mistral Chatbot")
chat_interface.launch()

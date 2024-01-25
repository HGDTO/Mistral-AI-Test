import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import pipeline

pipe = pipeline("text-generation", model="mistralai/Mistral-7B-v0.1")
# Load model directly
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")


# GPU 사용 설정 (가능한 경우)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 챗봇 기능을 수행하는 함수
def chat_with_mistral(user_input):
    inputs = tokenizer.encode(user_input, return_tensors="pt").to(device)
    outputs = model.generate(inputs, max_length=50, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Gradio 챗봇 인터페이스 설정
chat_interface = gr.Chatbot(
    fn=chat_with_mistral, 
    title="Mistral Chatbot",
    allow_flagging="never"
)

# 서버 시작
chat_interface.launch(server_name='0.0.0.0')

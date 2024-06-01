from flask import Flask, render_template, request
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    response = get_chat_response(msg)
    return response

def get_chat_response(text):
    # Generate a response based on the user input
    bot_input_ids = tokenizer.encode(text + tokenizer.eos_token, return_tensors='pt')
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    bot_response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    
    # Check for keywords related to healthcare topics and customize the response accordingly
    if any(keyword in text.lower() for keyword in ["who are you?", "who are you", "who r u?", "who r u", "help me", "who are u?", "who are u", "who r you?", "who r you"]):
        return "Hello! I am your virtual medical assistant, here to help with your health-related questions and provide information. While I can offer advice and guidance, please remember that I am not a substitute for professional medical care. For any serious concerns or emergencies, it is important to contact a healthcare provider directly. How can I assist you today?"
    
    elif any(keyword in text.lower() for keyword in ["symptoms", "symptom", "problem"]):
        return "Please provide more details about your symptoms."

    elif any(keyword in text.lower() for keyword in ["medication", "medicine", "drug"]):
        return "It's important to consult with a healthcare professional for advice on medication, as they can provide personalized recommendations based on your specific health needs."

    elif any(keyword in text.lower() for keyword in ["treatment", "therapy", "procedure"]):
        return "Treatment options vary depending on the condition. It's best to consult with a healthcare provider for personalized advice."

    elif any(keyword in text.lower() for keyword in ["diagnosis", "diagnose", "test"]):
        return "To get an accurate diagnosis, it's recommended to visit a healthcare professional who can perform necessary tests and assessments."

    elif any(keyword in text.lower() for keyword in ["prevention", "prevent", "protect"]):
        return "Preventive measures such as maintaining a healthy lifestyle, vaccination, and regular check-ups can help reduce the risk of certain diseases."

    elif any(keyword in text.lower() for keyword in ["diet", "nutrition", "food"]):
        return "A balanced diet rich in fruits, vegetables, lean proteins, and whole grains is essential for overall health and well-being."

    elif any(keyword in text.lower() for keyword in ["exercise", "physical activity", "workout"]):
        return "Regular exercise is important for maintaining cardiovascular health, muscle strength, and overall fitness."

    elif any(keyword in text.lower() for keyword in ["mental health", "stress", "anxiety"]):
        return "Taking care of your mental health is as important as your physical health. Consider activities like meditation, yoga, or talking to a therapist."

    elif any(keyword in text.lower() for keyword in ["vaccine", "vaccination", "immunization"]):
        return "Vaccination is one of the most effective ways to prevent infectious diseases and protect public health."

    elif any(keyword in text.lower() for keyword in ["covid-19", "coronavirus", "pandemic"]):
        return "Follow guidelines from health authorities, practice good hygiene, wear masks in crowded places, and get vaccinated to help prevent the spread of COVID-19."

    elif any(keyword in text.lower() for keyword in ["chronic disease", "chronic condition", "long-term illness"]):
        return "Managing chronic diseases requires a comprehensive approach involving medication, lifestyle changes, and regular monitoring by healthcare professionals."

    elif any(keyword in text.lower() for keyword in ["pregnancy", "maternity", "prenatal care"]):
        return "During pregnancy, it's important to receive regular prenatal care, eat a balanced diet, take prenatal vitamins, and avoid harmful substances."

    elif any(keyword in text.lower() for keyword in ["child health", "pediatrics", "childhood illness"]):
        return "Children should receive regular check-ups, vaccinations, and a healthy diet to support their growth and development."

    elif any(keyword in text.lower() for keyword in ["elderly care", "geriatrics", "aging"]):
        return "Elderly individuals may require specialized care, including regular medical check-ups, medication management, and support with daily activities."

    elif any(keyword in text.lower() for keyword in ["emergency", "urgent care", "accident"]):
        return "In case of emergencies or accidents, seek immediate medical attention by calling emergency services or visiting the nearest hospital."

    # Additional conditionals based on specific healthcare topics or queries
    elif any(keyword in text.lower() for keyword in ["allergy", "allergic reaction", "hay fever"]):
        return "Allergies can cause symptoms like sneezing, itching, or swelling. Avoiding triggers and using antihistamines can help manage symptoms."

    elif any(keyword in text.lower() for keyword in ["asthma", "respiratory condition", "wheezing"]):
        return "Asthma is a chronic condition that causes inflammation and narrowing of the airways. Inhalers and avoiding triggers can help manage symptoms."

    elif any(keyword in text.lower() for keyword in ["heart disease", "cardiovascular health", "high blood pressure"]):
        return "Heart disease refers to conditions that affect the heart's structure or function. Lifestyle changes, medication, and regular check-ups are important for managing heart health."

    elif any(keyword in text.lower() for keyword in ["stroke", "brain attack", "cerebrovascular accident"]):
        return "A stroke occurs when blood flow to the brain is interrupted. Prompt medical attention is crucial to minimize damage and improve outcomes."

    elif any(keyword in text.lower() for keyword in ["diabetes", "blood sugar", "insulin"]):
        return "Diabetes is a chronic condition that affects blood sugar levels. Monitoring blood sugar, medication, and lifestyle changes are key components of management."

    elif any(keyword in text.lower() for keyword in ["cancer", "tumor", "oncology"]):
        return "Cancer is a group of diseases characterized by abnormal cell growth. Treatment options include surgery, chemotherapy, radiation therapy, and immunotherapy."

    elif any(keyword in text.lower() for keyword in ["mental illness", "psychiatric disorder", "depression"]):
        return "Mental illnesses such as depression, anxiety, and bipolar disorder require treatment by mental health professionals. Therapy, medication, and support groups can help manage symptoms."

    elif any(keyword in text.lower() for keyword in ["substance abuse", "addiction", "drug addiction"]):
        return "Substance abuse and addiction can have serious consequences on physical and mental health. Treatment options include therapy, support groups, and medication-assisted therapy."

    elif any(keyword in text.lower() for keyword in ["nutritionist", "dietitian", "nutrition counseling"]):
        return "A nutritionist or dietitian can provide personalized dietary advice to help you achieve your health goals and manage medical conditions."

    elif any(keyword in text.lower() for keyword in ["physiotherapy", "physical therapy", "rehabilitation"]):
        return "Physiotherapy or physical therapy can help improve mobility, reduce pain, and promote recovery after injury or surgery."

    elif any(keyword in text.lower() for keyword in ["sleep disorder", "insomnia", "sleep apnea"]):
        return "Sleep disorders can affect your overall health and well-being. Lifestyle changes, therapy, and medication can help improve sleep quality."

    elif any(keyword in text.lower() for keyword in ["sexual health", "std", "contraception"]):
        return "Maintaining sexual health is important for overall well-being. Practice safe sex, get regular check-ups, and discuss any concerns with a healthcare provider."

    elif any(keyword in text.lower() for keyword in ["dermatology", "skin condition", "eczema"]):
        return "Dermatologists specialize in diagnosing and treating skin conditions. Proper skincare, medication, and lifestyle changes can help manage skin conditions."

    elif any(keyword in text.lower() for keyword in ["dentist", "dental health", "toothache"]):
        return "Regular dental check-ups, brushing, flossing, and a healthy diet are important for maintaining good oral health."

    elif any(keyword in text.lower() for keyword in ["gynecologist", "women's health", "menstrual cycle"]):
        return '''Regular gynecological exams, screenings, and preventive care are important for maintaining women's health. visit the "
        "<a href='https://www.example.com/womens-health'>Women's Health</a> page."'''

    elif any(keyword in text.lower() for keyword in ["urologist", "urinary tract infection", "prostate cancer"]):
        return "Urologists specialize in diagnosing and treating conditions of the urinary tract and male reproductive system. Regular check-ups and screenings are important for urological health."

    elif any(keyword in text.lower() for keyword in ["podiatry", "foot health", "plantar fasciitis"]):
        return "Podiatrists specialize in diagnosing and treating conditions of the feet and ankles. Proper footwear, orthotics, and exercises can help manage foot problems."

    elif any(keyword in text.lower() for keyword in ["physical health", "medical check-up", "wellness exam"]):
        return "Regular medical check-ups, screenings, and preventive care are important for maintaining overall physical health and well-being."

    # If no specific healthcare-related keyword is detected, return the bot's default response
    else:
        return bot_response


if __name__ == '__main__':
    app.run()

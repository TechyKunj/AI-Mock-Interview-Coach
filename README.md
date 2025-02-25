# 🎙 AI Mock Interview Coach

AI Interview Assistant is a Streamlit-based web application that helps users practice job interviews using AI-powered feedback. It generates interview questions based on a job description, allows users to respond via speech, transcribes their answers, and provides AI-driven analysis and feedback.

## 🚀 Features
- **Job-Specific Interview Questions**: Generates tailored interview questions based on a job description.
- **Speech-to-Text Processing**: Users can respond verbally, and the app transcribes their answers.
- **AI-Powered Answer Evaluation**: Uses Groq LLM to analyze responses and provide feedback.
- **Score and Feedback System**: Provides a numerical score and improvement tips for each answer.
- **Final Performance Report**: Summarizes overall performance at the end of the interview.

## 🛠 Tech Stack
- **Frontend**: Streamlit
- **Backend**: LangChain, Groq LLM (qwen-2.5-32b)
- **Database**: N/A (session state handling)
- **Speech Processing**: SpeechRecognition (Google Speech API)

## 📌 Installation
### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/yourusername/ai-interview-assistant.git
cd ai-interview-assistant
```

### **2️⃣ Set Up a Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate   # On macOS/Linux
venv\Scripts\activate     # On Windows
```

### **3️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

### **4️⃣ Add API Key (Groq LLM)**
Create a `.streamlit/secrets.toml` file and add:
```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

Alternatively, set the environment variable in your terminal:
```bash
export GROQ_API_KEY="your_groq_api_key_here"
```

## ▶️ Running the Application
```bash
streamlit run app.py
```

## 🖥 Deployment (Streamlit Cloud)
1. Push your project to GitHub.
2. Go to [Streamlit Cloud](https://share.streamlit.io/).
3. Deploy the repository and add `GROQ_API_KEY` under app settings > Secrets.
4. Restart the app if necessary.

## 🔥 Usage Guide
1. **Enter Job Description**: Provide details about the role you're preparing for.
2. **Start the Interview**: AI generates relevant questions.
3. **Answer via Microphone**: Speak your response; the app transcribes it.
4. **Get AI Feedback**: Receive a score and improvement suggestions.
5. **Review Final Report**: See an overall summary of your performance.

## 🤖 Future Improvements
- Support for more LLMs (OpenAI, Gemini, etc.).
- Saving session history for review.
- Multi-language support.

🚀 **Happy Interviewing!**


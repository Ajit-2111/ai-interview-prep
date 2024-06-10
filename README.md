
---

# Project Title: AI Interview Prep

## Introduction

Welcome to AI Interview Prep, an intelligent platform designed to help you prepare for job interviews using cutting-edge AI technologies. This platform integrates natural language processing (NLP), audio analysis, and computer vision to provide personalized feedback and guidance during interview practice sessions.

## File Structure

The project consists of the following components:

- **app.py**: The main Python script containing the Flask application for the web interface and coordinating interactions between different modules.
- **connectToOpenAI.py**: Python script responsible for connecting to OpenAI's API to generate interview questions and provide feedback on user answers.
- **audioEmotion.py**: Python script handling the detection of emotions from users' voices using audio analysis techniques.
- **textEmotion.py**: Python script implementing emotion detection from text inputs utilizing NLP models.
- **templates/**: Directory containing HTML templates for various pages of the web application.
- **static/**: Directory holding static assets like CSS, JavaScript, and image files.
- **models/**: Directory storing trained models used for emotion detection and other tasks.



Certainly! Here's a section describing the flow of code execution in the AI Interview Prep project:

---

## Code Flow

The AI Interview Prep project follows a structured flow of code execution to provide users with a seamless and interactive experience during interview preparation. Below is an overview of how the different components of the project interact with each other:

1. **Initialization**: 
   - The Flask application (`app.py`) is initialized, serving as the main entry point for the web interface.
   - Necessary modules and libraries are imported, including those responsible for handling web requests, connecting to external APIs (such as OpenAI), and performing audio and text analysis.

2. **User Interaction**:
   - Users interact with the web application by navigating through different pages and sections, such as interview practice, emotion analysis, and resources.

3. **Interview Practice**:
   - When users opt for interview practice, the application generates AI-generated interview questions using the `connectToOpenAI.py` module.
   - Users provide responses to these questions via text, voice or video inputs.

4. **Feedback Generation**:
   - The provided responses are analyzed for sentiment and emotional content using the `textEmotion.py` module for text inputs and the `audioEmotion.py` module for voice inputs.
   - Additionally, non-verbal communication cues are analyzed through face motion detection using functions in `app.py`.

5. **Integration of AI Models**:
   - Trained models stored in the `models/` directory are utilized for sentiment analysis, emotion detection, and face motion detection.
   - These models are loaded into memory as needed during runtime and used to process user inputs.

6. **Presentation of Results**:
   - The results of sentiment analysis, emotion detection, and face motion detection are aggregated and presented to the user in a user-friendly format.
   - Feedback on interview responses is provided based on the analysis results, helping users understand their strengths and areas for improvement.

7. **User Interaction and Navigation**:
   - Users can navigate between different sections of the application, access additional resources, and retake interview practice sessions as needed.

8. **Feedback and Improvement**:
   - Users can review feedback provided by the application, reflect on their performance, and use the insights gained to improve their interview skills over time.

9. **Continuous Development and Enhancement**:
   - The project undergoes continuous development and enhancement to incorporate new features, improve accuracy, and enhance user experience based on feedback and emerging technologies.

This flow ensures a seamless and interactive experience for users as they engage with the AI Interview Prep platform to enhance their interview preparation skills.

--- 

# Flask Application

---

## 1. Import Section

```python
import os
import cv2
from flask import Flask, render_template, request, redirect, jsonify
import base64
from keras.models import load_model
import numpy as np
from textEmotion import textSentiment
from audioEmotion import model_test
from connectToOpenAI import getQuestions, getSuggestions
from copy import deepcopy
```

This section imports the necessary libraries and modules required for the functionality of the application. The imported modules include functions and classes for image processing, web development using Flask, deep learning model loading, numerical computing, sentiment analysis, audio emotion analysis, and interfacing with OpenAI for question generation.

---

## 2. Global Variables Section

```python
UPLOAD_FOLDER = 'uploads'
personalDetail = {}
interviewFullResponse = []
questions = None
questionsCopy = None
personalDetailCopy = None
noOfQuestions = '2'
emotionCount = {'Angry': 0, 'Neutral': 0, 'Fearful': 0, 'Happy': 0, 'Sad': 0, 'Surprise': 0}
```

This section declares global variables used throughout the application. These variables include paths for file upload, storage for personal details, interview responses, interview questions, and their copies, the number of questions to be asked, and a dictionary to store emotion counts.

---

## 3. Function: detectEmotion

```python
def detectEmotion(imgPath: str):
    """
    Detects emotions from a given image.

    Parameters:
        imgPath (str): The path to the image file.

    Returns:
        None
    """
    img1 = cv2.imread(imgPath)
    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    faces = cascade.detectMultiScale(gray, 1.1, 3)

    for x, y, w, h in faces:
        cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cropped = img1[y:y + h, x:x + w]

    cv2.imwrite('static/after.jpg', img1)

    cv2.imwrite('static/cropped.jpg', cropped)
    image = cv2.imread('static/cropped.jpg', 0)
    image = cv2.resize(image, (48, 48))

    image = image / 255.0

    image = np.reshape(image, (1, 48, 48, 1))

    model = load_model('models/model_3.h5')

    prediction = model.predict(image)

    label_map = ['Angry', 'Neutral', 'Fearful', 'Happy', 'Sad', 'Surprise']

    prediction = np.argmax(prediction)

    final_prediction = label_map[prediction]
    print(final_prediction)
    emotionCount[final_prediction] = emotionCount.get(final_prediction) + 1
    print(emotionCount)
    os.remove('static/after.jpg')
    os.remove('static/cropped.jpg')
```

This function detects emotions from a given image. It reads the image using OpenCV, detects faces, preprocesses the face images, predicts emotions using a pre-trained deep learning model, updates emotion counts, and removes temporary files generated during the process.

---

## 4. Function: appendInterviewResponse

```python
def appendInterviewResponse(question: str, answer: str, suggestedAnswer: str, feedback: str, sentiment: str):
    """
    Appends interview responses to the interviewFullResponse list.

    Parameters:
        question (str): The interview question.
        answer (str): The candidate's answer.
        suggestedAnswer (str): A suggested answer for the question.
        feedback (str): Feedback on the candidate's answer.
        sentiment (str): Sentiment analysis result of the candidate's answer.

    Returns:
        None
    """
    interviewFullResponse.append(
        {'question': question, 'answer': answer, 'suggestedAnswer': suggestedAnswer, 'feedback': feedback,
         'sentiment': sentiment})
```

This function appends interview responses to the `interviewFullResponse` list. It takes the question, answer, suggested answer, feedback, and sentiment as input and appends them as a dictionary to the list.

---

## 5. Flask Routes Section

```python
# Flask routes for different pages and functionalities

@app.route("/", methods=["GET"])
def home():
    return render_template("landing.html")


@app.route("/about", methods=["GET"])
def about():
    return render_template("about.html")


@app.route("/personal-details", methods=["GET"])
def personal_details():
    global questionsCopy, interviewFullResponse, personalDetailCopy, questions
    print("inside personalDetail", personalDetail)
    questionsCopy = None
    questions = None
    personalDetailCopy = {}
    interviewFullResponse = []
    return render_template("personal-details.html")
```

This section contains various Flask routes for rendering HTML templates and handling different functionalities of the application. The routes include:
- `/`: Home page route.
- `/about`: About page route.
- `/personal-details`: Personal details page route.

Each route corresponds to a specific URL endpoint and HTTP method. Upon receiving a request, the associated function renders an HTML template using the `render_template` function.

---

## 6. Function: read_personal_details

```python
@app.route("/read_personal_details", methods=["POST", "GET"])
def read_personal_details():
    global personalDetail, questions, noOfQuestions, personalDetailCopy, questionsCopy
    data = request.form
    questions = getQuestions(domain_name=data['job-role'], type_of_interview=data['interview_type'],
                             work_experience=data['experience_level'], no_of_question=noOfQuestions)
    questionsCopy = deepcopy(questions)
    personalDetail = {
        'interview_type': data['interview_type'],
        'experience_level': data['experience_level'],
        'job-role': data['job-role'],
    }
    personalDetailCopy = deepcopy(personalDetail)
    return redirect("/interview-question")
```

This function is a route handler for processing personal details form data submitted by the user. It extracts form data using `request.form`, fetches interview questions based on the provided details, and initializes global variables accordingly. Finally, it redirects the user to the interview question page.

---


## 7. Function: interview_question

```python
@app.route("/interview-question", methods=["GET"])
def interview_question():
    global questions, personalDetail, interviewFullResponse
    pageHeading = personalDetail["job-role"].capitalize()
    if not questions:
        return redirect("/interview-analysis")
    pageQuestion = questions.pop(0)
    return render_template("interview-question.html", pageQuestion=pageQuestion, pageHeading=pageHeading)
```

This function serves as a route handler for displaying interview questions to the user. It retrieves the next question from the `questions` list, updates the page heading based on the job role, and renders the interview question HTML template.

---

## 8. Function: interview_response

```python
@app.route("/interview-response", methods=["POST"])
def interview_response():
    data = request.form
    sentiment = textSentiment(data['answer'])
    feedback, suggestedAnswer = getSuggestions(question=data['curQuestion'], answer=data['answer'])
    appendInterviewResponse(data['curQuestion'], data['answer'], suggestedAnswer, feedback, sentiment)
    return redirect("/interview-question")
```

This function handles the submission of interview responses by the user. It extracts the response data from the form, performs sentiment analysis on the answer, retrieves feedback and suggested answers, and appends the response to the `interviewFullResponse` list. Finally, it redirects the user to the next interview question.

---

## 9. Function: interview_analysis

```python
@app.route("/interview-analysis", methods=["GET"])
def interview_analysis():
    global interviewFullResponse
    return render_template("interviewAnalysis.html", interviewFullResponse=interviewFullResponse)
```

This function serves as a route handler for displaying the analysis of interview responses. It renders the interview analysis HTML template, passing the `interviewFullResponse` list as context data to be displayed.

---

## 10. Function: retake_test

```python
@app.route("/retake-test", methods=["GET"])
def retake_test():
    global questions, interviewFullResponse, personalDetail
    questions = deepcopy(questionsCopy)
    personalDetail = deepcopy(personalDetailCopy)
    interviewFullResponse = []
    return redirect("/interview-question")
```

This function handles the request for retaking the interview test. It resets the interview-related variables (`questions`, `interviewFullResponse`) to their initial states by making deep copies of the original values stored in `questionsCopy` and `personalDetailCopy`. Then, it redirects the user to the interview question page to start the test again.

---

## 11. Function: image_upload

```python
@app.route('/image-upload', methods=['POST'])
def image_upload():
    try:
        request_data = request.get_json()
        image_data_b64 = request_data['image']

        # Decode base64-encoded image data
        image_data = base64.b64decode(image_data_b64.split(",")[1])

        # Process the image data (e.g., save to disk, analyze, etc.)
        # Here, we're just printing the size of the image data for demonstration
        print('Received image with size:', len(image_data))
        image_path = os.path.join(UPLOAD_FOLDER, 'uploaded_image.jpg')
        with open(image_path, 'wb') as f:
            f.write(image_data)
        detectEmotion(os.path.join(UPLOAD_FOLDER, 'uploaded_image.jpg'))
        os.remove(os.path.join(UPLOAD_FOLDER, 'uploaded_image.jpg'))
        return '', 200
    except Exception as e:
        print('Error processing image:', e)
        return 'Error processing image', 500
```

This function serves as an endpoint for uploading images and processing them to detect emotions. It extracts the base64-encoded image data from the request, decodes it, and processes the image using the `detectEmotion` function. Finally, it returns an appropriate HTTP response.

---


## 12. Function: audioemotion

```python
@app.route("/audioemotion", methods=['GET', 'POST'])
def audioemotion():
    if request.method == "POST":
        f = request.files['audio_data']
        results = model_test(f)
        print('audio file uploaded successfully****')
        print(results)
        jsonresults = {
            "audioemotion": results["predictedEmotion"],
            "audioPS": results["predictedSex"],
        }
        return jsonify(jsonresults)
```

This function serves as an endpoint for analyzing audio emotions. It handles both GET and POST requests. When receiving a POST request, it expects audio data in the request files under the key 'audio_data'. It then passes this data to the `model_test` function for analysis. Once the analysis is done, it constructs a JSON response containing the predicted emotion and predicted sex (if applicable) and returns it to the client.

---

## 13. Main Execution Section

```python
if __name__ == "__main__":
    app.run(debug=True)
```

This section checks if the script is being run as the main program, and if so, it starts the Flask application in debug mode. Running the application in debug mode enables additional debugging information and automatic reloading of the application when changes are made to the source code.

---




# Connect to OpenAI code.

This article outlines the structure and purpose of each section and function in the provided code for interacting with the GPT-3.5 model using the OpenAI Python SDK.

---

## 1. Import Section

```python
from openai import OpenAI
import json
```

This section imports the necessary libraries and modules required for the functionality of the code. The `OpenAI` module is imported from the OpenAI Python SDK for interfacing with the GPT-3.5 model, and the `json` module is imported for JSON serialization and deserialization.

---

## 2. Initialization of OpenAI Client

```python
client = OpenAI(api_key="ENTER YOUR API KEY")
```

This line initializes an instance of the OpenAI client using your API key. Replace `"ENTER YOUR API KEY"` with your actual API key obtained from the OpenAI website.

---

## 3. Function: generate_response

```python
def generate_response(prompt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content
```

This function generates a response based on the provided prompt using the GPT-3.5 model. It sends the prompt to the GPT-3.5 model and receives a completion containing the response. The response is extracted from the completion and returned.

---

## 4. Function: getQuestions

```python
def getQuestions(domain_name: str, type_of_interview: str, work_experience: str, no_of_question: str):
    promptforQuestion = f"Give me a single list with json key as 'questions' {no_of_question} {type_of_interview} interview questions for a {work_experience} {domain_name}"
    aiResponse = generate_response(promptforQuestion)
    jsonquestions = json.loads(aiResponse)
    return jsonquestions['questions']
```

This function generates interview questions based on the provided parameters using the GPT-3.5 model. It constructs a prompt asking for a list of interview questions and sends it to the GPT-3.5 model. The response containing the generated questions is then parsed from JSON format and returned.

---

## 5. Function: getSuggestions

```python
def getSuggestions(question: str, answer: str):
    promptForAnswer = f"Give me your response in key named 'feedback' and 'suggestedAnswer' and the value must me string for this question '{question}' how is this answer: {answer} ? If there are any suggestion please provide it."
    aiResponse = generate_response(promptForAnswer)
    jsonquestions = json.loads(aiResponse)
    return jsonquestions['feedback'], jsonquestions['suggestedAnswer']
```

This function generates feedback and suggested answers for a given question and answer pair using the GPT-3.5 model. It constructs a prompt asking for feedback and suggested answers based on the provided question and answer. The response containing the feedback and suggested answer is then parsed from JSON format and returned.

---

# Text Emotion Recognition

This article outlines the structure and purpose of each section and function in the provided code for text preprocessing and sentiment analysis.

---

## 1. Import Section


```python
import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
```

This section imports the necessary libraries and modules required for text preprocessing and sentiment analysis. It includes modules for regular expressions, Natural Language Toolkit (NLTK) for text processing, NumPy for numerical operations, and Keras for loading the sentiment analysis model.

---

## 2. Setup Section

```python
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words("english"))
```

This section initializes NLTK and downloads necessary resources like wordnet, stopwords, and Open Multilingual Wordnet (OMW). It also sets up a set of English stopwords.

---

## 3. Text Preprocessing Functions

This section contains several functions for preprocessing text data:

- **lemmatization**: Performs lemmatization on the input text.
- **remove_stop_words**: Removes stopwords from the input text.
- **Removing_numbers**: Removes numbers from the input text.
- **lower_case**: Converts the input text to lowercase.
- **Removing_punctuations**: Removes punctuations from the input text.
- **Removing_urls**: Removes URLs from the input text.
- **remove_small_sentences**: Removes sentences with less than 3 words from the input DataFrame.
- **normalize_text**: Applies a series of text preprocessing steps to the input DataFrame.
- **normalized_sentence**: Normalizes a single input sentence by applying text preprocessing steps.

---

## 4. Loading Models

```python
with open('./models/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('./models/label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

model = load_model('./models/Emotion Recognition From English text.h5')
```

This section loads the tokenizer, label encoder, and sentiment analysis model from disk. These models are used for preprocessing text and predicting sentiment.

---

## 5. Sentiment Analysis Function

```python
def textSentiment(sentence: str):
    print(sentence)
    sentence = normalized_sentence(sentence)
    sentence = tokenizer.texts_to_sequences([sentence])
    sentence = pad_sequences(sentence, maxlen=229, truncating='pre')
    result = le.inverse_transform(np.argmax(model.predict(sentence), axis=-1))[0]
    proba = np.max(model.predict(sentence))
    print(f"{result} : {proba}\n\n")
    return result
```

This function performs sentiment analysis on a given input sentence. It preprocesses the input sentence, converts it into a sequence of tokens, pads the sequence to a fixed length, predicts the sentiment using the loaded model, and returns the predicted sentiment label.

---

## 6. Example Usage

```python
textSentiment("hello how are you ?")
```

This line demonstrates how to use the `textSentiment` function by passing a sample input sentence. The function prints the input sentence, predicts the sentiment, prints the predicted sentiment label along with its probability, and returns the predicted sentiment label.

---

# Audio Emotion Recognition
This article outlines the structure and purpose of each section and function in the provided code for audio processing and emotion/gender prediction.

---

## 1. Import Section

```python
from joblib import load
import librosa
import numpy as np
```

This section imports the necessary libraries and modules required for audio processing and model loading. It includes modules for loading models with joblib, audio processing with librosa, and numerical operations with NumPy.

---

## 2. Global Variables

```python
results_dict = {
    "predictedEmotion": [],
    "emotionCategories": [],
    "probabilities": [],
    "predictedSex": []
}

user_file = {
    'filepath': []
}
```

This section declares global dictionaries `results_dict` and `user_file` used for storing the results of the emotion and gender prediction, as well as the filepath of the user's audio file, respectively.

---

## 3. Function: input_parser

```python
def input_parser(input_file):
    try:
        X, sample_rate = librosa.load(input_file, res_type='kaiser_fast')
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=128).T, axis=0)
    except Exception as e:
        print("Error encountered while parsing file: ", e)
        return None
    feature = mfccs.tolist()

    return feature
```

This function parses the input audio file and extracts features for model input. It loads the audio file using librosa, calculates Mel-frequency cepstral coefficients (MFCCs) as features, and returns the features as a list.

---

## 4. Function: model_test

```python
def model_test(input_file):
    user_file["filepath"] = input_file
    print("*****Audio Received******")
    model = load('models/rf2_model.sav')
    model2 = load('models/gen_emo_rf_model.sav')
    feature = input_parser(input_file)
    arr = np.array(feature)
    arr2d = np.reshape(arr, (1, 128))
    pred_emotion = model.predict(arr2d)
    probs = model.predict_proba(arr2d)
    emotion_labels = model.classes_
    gender = model2.predict(arr2d)
    if gender[0] == 0:
        label = "Male"
    elif gender[0] == 1:
        label = "Female"
    results_dict["predictedEmotion"] = pred_emotion[0]
    results_dict["emotionCategories"] = emotion_labels.tolist()
    results_dict["probabilities"] = probs[0].tolist()
    results_dict["predictedSex"] = label
    print(results_dict)
    return results_dict
```

This function performs emotion and gender prediction on the input audio file. It loads pre-trained models using joblib, extracts features from the input audio using the `input_parser` function, and makes predictions using the models. The results are stored in the `results_dict` dictionary and returned.

---


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

app = Flask(__name__)

personalDetail = {}
interviewFullResponse = []
questions = None
questionsCopy = None
personalDetailCopy = None
noOfQuestions = '2'
emotionCount = {'Angry': 0, 'Neutral': 0, 'Fearful': 0, 'Happy': 0, 'Sad': 0, 'Surprise': 0}

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


def detectEmotion(imgPath: str):
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


def appendInterviewResponse(question: str, answer: str, suggestedAnswer: str, feedback: str, sentiment: str):
    global interviewFullResponse
    interviewFullResponse.append(
        {'question': question, 'answer': answer, 'suggestedAnswer': suggestedAnswer, 'feedback': feedback,
         'sentiment': sentiment})
    print(interviewFullResponse)


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


@app.route("/interview-question", methods=["GET"])
def interview_question():
    global questions, personalDetail, interviewFullResponse
    print(personalDetail)
    pageHeading = personalDetail["job-role"].capitalize()
    if not questions:
        return redirect("/interview-analysis")
    pageQuestion = questions.pop(0)
    return render_template("interview-question.html", pageQuestion=pageQuestion, pageHeading=pageHeading)


@app.route("/interview-response", methods=["POST"])
def interview_response():
    data = request.form
    sentiment = textSentiment(data['answer'])
    feedback, suggestedAnswer = getSuggestions(question=data['curQuestion'], answer=data['answer'])
    appendInterviewResponse(data['curQuestion'], data['answer'], suggestedAnswer, feedback, sentiment)
    return redirect("/interview-question")


@app.route("/interview-analysis", methods=["GET"])
def interview_analysis():
    global interviewFullResponse
    return render_template("interviewAnalysis.html", interviewFullResponse=interviewFullResponse)


@app.route("/retake-test", methods=["GET"])
def retake_test():
    global questions, interviewFullResponse, personalDetail
    print("inside before retake", personalDetail, questions)
    questions = deepcopy(questionsCopy)
    personalDetail = deepcopy(personalDetailCopy)
    interviewFullResponse = []
    print("inside after retake", personalDetail, questions)
    return redirect("/interview-question")


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

# *************AUDIO PAGE******************
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

# *************END AUDIO PAGE******************

if __name__ == "__main__":
    app.run(debug=True)

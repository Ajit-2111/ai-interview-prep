from joblib import load
import librosa
import numpy as np


results_dict = {
    "predictedEmotion": [],
    "emotionCategories": [],
    "probabilities": [],
    "predictedSex": []
}

user_file = {
    'filepath': []
}


def input_parser(input_file):
    try:
        X, sample_rate = librosa.load(input_file, res_type='kaiser_fast')
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=128).T, axis=0)
    except Exception as e:
        print("Error encountered while parsing file: ", e)
        return None
    feature = mfccs.tolist()

    return feature


def model_test(input_file):
    user_file["filepath"] = input_file
    print("*****Audio Recieved******")
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

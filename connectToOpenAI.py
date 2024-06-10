from openai import OpenAI
import json

client = OpenAI(api_key="ENTER YOUR API KEY")


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


def getQuestions(domain_name: str, type_of_interview: str, work_experience: str, no_of_question: str):
    promptforQuestion = f"Give me a single list with json key as 'questions' {no_of_question} {type_of_interview} interview questions for a {work_experience} {domain_name}"
    aiResponse = generate_response(promptforQuestion)
    jsonquestions = json.loads(aiResponse)
    print(jsonquestions['questions'])
    return jsonquestions['questions']


def getSuggestions(question: str, answer: str):
    promptForAnswer = f"Give me your response in key named 'feedback' and 'suggestedAnswer' and the value must me string for this question '{question}' how is this answer: {answer} ? If there are any suggestion please provide it."
    aiResponse = generate_response(promptForAnswer)
    print(aiResponse)
    jsonquestions = json.loads(aiResponse)
    print(jsonquestions['suggestedAnswer'])
    print(jsonquestions['feedback'])
    return jsonquestions['feedback'], jsonquestions['suggestedAnswer']


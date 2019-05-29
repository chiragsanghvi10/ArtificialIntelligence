from flask import Flask
from flask import request
from flask import jsonify
from flask import send_from_directory
from text import sentence_similarity as sentence_similarity_api
from speech.analysis import main as analysis_api
from speech.emotion import emotion_api
from speech.utils import misc
import jsonpickle
import redis
from speech.transcription.transfer_learning import chunk_data_api as chunk_api
import tensorflow as tf
import tensorflow_hub as hub
import json
import uuid
from text import bert_mrpc as bert_api

app = Flask(__name__)
loaded_model = None
pool = redis.ConnectionPool(host='localhost', port=6379, db=0)
embed = g = session = messages = output = None

@app.route("/sentence_similarity", methods=['GET', 'POST'])
def sentence_similarity():
    global embed,g,session,messages,output
    if g == None:
        module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
        embed = hub.Module(module_url)
        g = tf.get_default_graph()
        session = tf.Session(graph=g)
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        messages = tf.placeholder(dtype=tf.string, shape=[None])
        output = embed(messages)
        print('Successfully initialized sentence similarity variables')
    sentence1 = request.form['sentence1']
    sentence2 = request.form['sentence2']
    return sentence_similarity_api.fast_sentence_similarity(sentence1, sentence2, g, output, session, messages)

@app.route('/sentence_similarity_bert', methods=['GET', 'POST'])
def sentence_similarity_bert():
    sentence1 = request.form["sentence1"]
    sentence2 = request.form["sentence2"]
    semantic_similarity = bert_api.predict(sentence1, sentence2)
    print('Sentence1: {}\n Sentence2: {}\nSimilarity: {}'.format(sentence1, sentence2, semantic_similarity['similarity']))
    return jsonpickle.encode(semantic_similarity)

@app.route("/transcibe", methods=['GET', 'POST'])
def transcibe():
    task_id = request.args['task_id']
    if task_id is None:
        task_id = uuid.uuid4().hex
    task_url = misc.get_task_url(task_id)
    url = request.args.get("url")
    if url is None:
        print("URL not provided")
    else:
        task_url = url
    language = request.args['language']
    model = (request.args['model'] == 'True')
    engine = request.args['engine']
    if engine is None:
        engine = 'google'
    print('Started: '+task_id)
    conversation_blocks = analysis_api.transcribe_emotion(engine, task_id, language, model, loaded_model, pool, task_url, False)
    print('Finished: '+task_id)
    return jsonpickle.encode(conversation_blocks)

@app.route("/transcibe_emotion", methods=['GET', 'POST'])
def transcibe_emotion():
    task_id = request.args['task_id']
    if task_id is None:
        task_id = uuid.uuid4().hex
    task_url = misc.get_task_url(task_id)
    url = request.args.get("url")
    if url is None:
        print("URL not provided")
    else:
        task_url = url
    language = request.args['language']
    model = (request.args['model'] == 'True')
    engine = request.args['engine']
    if engine is None:
        engine = 'google'
    conversation_blocks = analysis_api.transcribe_emotion(engine, task_id, language, model, loaded_model, pool, task_url)
    return jsonpickle.encode(conversation_blocks)

@app.route("/emotion", methods=['GET', 'POST'])
def emotion():
    task_id = request.args['task_id']
    if task_id is None:
        task_id = uuid.uuid4().hex
    url = request.args.get("url")
    if url is None:
        print("URL not provided")
        task_url = misc.get_task_url(task_id)
    else:
        task_url = url
    global loaded_model
    if loaded_model is None:
        loaded_model = emotion_api.getModel()
        loaded_model._make_predict_function()
    emotion_blocks = analysis_api.emotion(task_url, task_id, loaded_model)
    return jsonpickle.encode(emotion_blocks)

@app.route("/emotion_stream", methods=['GET', 'POST'])
def emotion_stream():
    task_id = request.args.get("task_id")
    url = request.args.get("url")
    global loaded_model
    if loaded_model is None:
        loaded_model = emotion_api.getModel()
        loaded_model._make_predict_function()
    emotion_blocks = emotion_api.windowing_emotion(url, task_id, loaded_model)
    return jsonpickle.encode(emotion_blocks)

@app.route("/chunks", methods=['GET', 'POST'])
def chunks():
    page = request.args['page']
    pagination = request.args['pagination']
    chunks = chunk_api.fetch_chunks(page, pagination)
    return jsonpickle.encode(chunks)

@app.route("/verify_chunk", methods=['GET', 'POST'])
def verify_chunk():
    chunk_id = request.args['chunk_id']
    is_verified = request.args['is_verified'].lower().startswith('t')
    chunks = chunk_api.mark_chunk_as_verified(chunk_id, is_verified)
    return jsonpickle.encode(chunks)

@app.route("/seen_chunk", methods=['GET', 'POST'])
def seen_chunk():
    chunk_id = request.args['chunk_id']
    chunks = chunk_api.mark_chunk_as_seen(chunk_id)
    return jsonpickle.encode(chunks)

@app.route("/update_chunk_transcription", methods=['GET', 'POST'])
def update_chunk_transcription():
    chunk_id = request.args['chunk_id']
    transcript = request.args['transcript']
    print("New: "+transcript)
    chunks = chunk_api.update_chunk_transcription(chunk_id, transcript)
    return jsonpickle.encode(chunks)

@app.route('/')
def send_base():
    path = 'index.html'
    return send_from_directory('static/', path)

@app.route('/verified_chunks', methods=['GET'])
def send_verified_static():
    path = 'index2.html'
    return send_from_directory('static/', path)

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

@app.route('/audio/<path:path>')
def send_static_audio(path):
    return send_from_directory('/home/absin/Downloads/dataset/chunks', path)

@app.route('/api/snippets/<int:count>/<int:page>')
def serve_snippet(count, page):
    data = chunk_api.fetch_verified_chunks(count, page)
    return json.dumps({'response': data})

if __name__ == '__main__':
    app.run(debug=True, threaded=True, host='0.0.0.0', port='5010')

import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from utils import vad as vad_api
from utils import misc
from utils import objects
from utils import constants
import librosa
import numpy as np
import pandas as pd
import os
from keras.models import model_from_json
import jsonpickle
import shutil
from flask import Flask
from flask import request
import time
import tensorflow as tf
import webrtcvad

def emotion(channel_files, loaded_model, task_folder, task_id):
    tf.keras.backend.clear_session()
    folder = task_folder + 'emotion_chunks/' + task_id+'/'
    if os.path.exists(folder) and os.path.isdir(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)
    # Download the task audio
    agent_snippets = vad_api.perform_vad(channel_files[0], folder)
    cust_snippets = vad_api.perform_vad(channel_files[1], folder)
    for snippet in agent_snippets:
        predicted, score = getEmotionPredictionChunk(
            snippet.path, loaded_model)
        snippet.add_signal(objects.Signal(predicted, score))
        snippet.set_speaker('Agent')
        if float(score) > 0.6:
            print('Agent: For time: ' + str(snippet.from_time) + ' to time: ' +
                  str(snippet.to_time) + ' Predicted label - > ' + predicted + ' with score -> ' + score + ' on file: ' + snippet.path)
    for snippet in cust_snippets:
        predicted, score = getEmotionPredictionChunk(
            snippet.path, loaded_model)
        snippet.add_signal(objects.Signal(predicted, score))
        snippet.set_speaker('Customer')
        if float(score) > 0.6:
            print('Customer: For time: ' + str(snippet.from_time) + ' to time: ' +
                  str(snippet.to_time) + ' Predicted label - > ' + predicted + ' with score -> ' + score + ' on file: ' + snippet.path)
    snips = agent_snippets + cust_snippets
    snips.sort(key=lambda x: x.from_time, reverse=False)
    #shutil.rmtree(folder)
    return snips


def getModel(model_path="NA", weight_path = "NA"):
    if model_path == "NA":
        model_path = constants.fetch_contant('emotion', 'model_path')
    if weight_path == "NA":
        weight_path = constants.fetch_contant('emotion', 'weight_path')
    json_file = open(model_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(weight_path)
    print("Loaded model and weights from disk")
    return loaded_model


def getEmotionPredictionChunk(f, loaded_model):
    """Get emotional score from a 2 second chunk file"""
    labels = ['female_angry', 'female_calm', 'female_fearful', 'female_happy',
              'female_sad', 'male_angry', 'male_calm', 'male_fearful', 'male_happy', 'male_sad']
    X, sample_rate = librosa.load(
        f, res_type='kaiser_fast', duration=2.5, sr=22050*2, offset=0)
    sample_rate = np.array(sample_rate)
    mfccs = np.mean(librosa.feature.mfcc(
        y=X, sr=sample_rate, n_mfcc=13), axis=0)
    featurelive = mfccs
    livedf2 = featurelive
    livedf2 = pd.DataFrame(data=livedf2)
    livedf2 = livedf2.stack().to_frame().T
    twodim = np.expand_dims(livedf2, axis=2)
    livepreds = loaded_model.predict(twodim,
                                     batch_size=32,
                                     verbose=1)
    livepreds1 = livepreds.argmax(axis=1)
    return labels[livepreds1[0]], str(livepreds[0][livepreds1[0]])

def is_valid_chunk(chunks, sample_rate11):
    aggressiveness = 3
    vad = webrtcvad.Vad(aggressiveness)
    yes_count = 0
    no_count = 0
    frame_dur_ms = 20
    window_chunks = int(frame_dur_ms*sample_rate11/1000)
    curr_ind = 0.0
    has_more = (len(chunks)>window_chunks)
    from_time = 0.0
    to_time = frame_dur_ms/1000
    while(has_more):
        vad_input = chunks[int(curr_ind):int(curr_ind+window_chunks)]
        assert webrtcvad.valid_rate_and_frame_length(sample_rate11, len(vad_input))
        try:
            is_speech = vad.is_speech(vad_input, sample_rate11)
        except Exception as e:
            print(e)
            is_speech = False
        if is_speech:
            yes_count += 1
        else:
            no_count += 1
        #print('{} - {} : {}'.format(from_time, to_time, is_speech))
        curr_ind += window_chunks
        from_time += frame_dur_ms/1000
        to_time += frame_dur_ms/1000
        has_more = (len(chunks)>curr_ind)
    print('{} : {}'.format(yes_count, no_count))
    if ((yes_count*100/(yes_count + no_count)) > 80):
        return True
    else:
        return False

def windowing_emotion(url, task_id, loaded_model):
    snippets = []
    tf.keras.backend.clear_session()
    task_folder = '/home/absin/git/sentenceSimilarity/speech/audio/tasks/'
    if url is not None:
        f = misc.download_file(url, task_folder)
    if task_id is not None:
        f = misc.download_file(misc.get_task_url(task_id), task_folder)
    X, sample_rate = librosa.load(f['abs_path'], res_type='kaiser_fast', sr=22050*2, offset=0)
    sample_rate = np.array(sample_rate)
    num_samples = X.shape[0]
    time = num_samples / (44100)
    time
    chunk_duration  = 2.5
    chunk_window = 0.5
    chunk_frames = int(44100 * chunk_duration)
    chunk_window_frames = int(44100 * chunk_window)
    # We need to collect samples every 2.5 second window and jump by say 0.5 seconds
    # In order to do this we will take 2.5* 44100 from the X and make it into a vector then pass it to the model
    down_sampled = f['abs_path'].replace('.wav', '_mono.wav')
    misc.stereo_to_mono(f['abs_path'], down_sampled)
    X11, sample_rate11, audio_length = vad_api.read_wave(down_sampled)
    curr_index = 0
    from_time = 0.0
    to_time = 2.5
    has_elements = (len(X)>chunk_frames)
    while(has_elements):
        chunk = X[curr_index:(curr_index+chunk_frames)]
        snippet = objects.Snippet(f['abs_path'], from_time, to_time)
        mfccs = np.mean(librosa.feature.mfcc(
            y=chunk, sr=sample_rate, n_mfcc=13), axis=0)
        featurelive = mfccs
        livedf2 = featurelive
        livedf2 = pd.DataFrame(data=livedf2)
        livedf2 = livedf2.stack().to_frame().T
        twodim = np.expand_dims(livedf2, axis=2)
        livepreds = loaded_model.predict(twodim,batch_size=32,verbose=1)
        livepreds1 = livepreds.argmax(axis=1)
        labels = ['female_angry', 'female_calm', 'female_fearful', 'female_happy',
                  'female_sad', 'male_angry', 'male_calm', 'male_fearful', 'male_happy', 'male_sad']
        ppp = labels[livepreds1[0]] + str(livepreds[0][livepreds1[0]])
        if is_valid_chunk(X11[int(from_time*sample_rate11):int(to_time*sample_rate11)], sample_rate11):
            if livepreds[0][livepreds1[0]] > 0.85:
                print('{} -- {} ---> {}'.format(from_time, to_time, ppp))
                snippet.add_signal(objects.Signal(labels[livepreds1[0]], str(livepreds[0][livepreds1[0]])))
            else:
                print('{} -- {} ---> Invalid'.format(from_time, to_time))
        if len(snippet.signals)>0:
            snippets.append(snippet)
        curr_index += chunk_window_frames
        from_time+=chunk_window
        to_time+=chunk_window
        has_elements = ((curr_index + chunk_frames) < len(X))
    return snippets









if __name__ == '__main__':
    loaded_model = getModel("/home/absin/Documents/dev/sentenceSimilarity/speech/emotion/models/model.json",
    "/home/absin/Documents/dev/sentenceSimilarity/speech/emotion/models/Emotion_Voice_Detection_Model.h5")
    loaded_model._make_predict_function()
    print(getEmotionPredictionChunk('/home/absin/Downloads/output.wav',loaded_model))

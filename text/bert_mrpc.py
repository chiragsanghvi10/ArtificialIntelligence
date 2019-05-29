import tensorflow as tf
import time
import os
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

from lib.bert import  run_classifier
from lib.bert import  modeling
from lib.bert import  optimization
from lib.bert import  tokenization


BERT_PRETRAINED_DIR='/home/absin/Documents/dev/bert/model/msrc/'
CONFIG_FILE = os.path.join(BERT_PRETRAINED_DIR, 'bert_config.json')
VOCAB_FILE = os.path.join(BERT_PRETRAINED_DIR, 'vocab.txt')
#INIT_CHECKPOINT = os.path.join(BERT_PRETRAINED_DIR, 'bert_model.ckpt')
INIT_CHECKPOINT = os.path.join(BERT_PRETRAINED_DIR, 'bert-checkpoints_models_MRPC_model.ckpt-343')

OUTPUT_DIR=BERT_PRETRAINED_DIR
SAVE_CHECKPOINTS_STEPS=99999999
ITERATIONS_PER_LOOP=1000
NUM_TPU_CORES=8
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 8
PREDICT_BATCH_SIZE = 1
LEARNING_RATE=2e-5
NUM_TRAIN_EPOCHS = 3.0
MAX_SEQ_LENGTH = 128
# Warmup is a period of time where hte learning rate
# is small and gradually increases--usually helps training.
WARMUP_PROPORTION = 0.1
# Model configs
SAVE_CHECKPOINTS_STEPS = 1000
SAVE_SUMMARY_STEPS = 500
processor=run_classifier.MrpcProcessor()
label_list=processor.get_labels()
#train_examples = processor.get_train_examples(TASK_DATA_DIR)
train_examples=range(10000)
num_train_steps = int(len(train_examples) / TRAIN_BATCH_SIZE * NUM_TRAIN_EPOCHS)
num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)
tokenizer = tokenization.FullTokenizer(vocab_file=VOCAB_FILE, do_lower_case=True)
model_fn =  estimator_from_checkpoints = None

def get_run_config(output_dir):
    tpu_cluster_resolver = None
    return tf.contrib.tpu.RunConfig(
    cluster=tpu_cluster_resolver,
    master=None,
    model_dir=output_dir,
    save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS,
    tpu_config=tf.contrib.tpu.TPUConfig(
        iterations_per_loop=ITERATIONS_PER_LOOP,
        num_shards=NUM_TPU_CORES,
        per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2))


def init():
    """Call this method once to initialize the graph and then keep reusing"""
    global model_fn, estimator_from_checkpoints
    model_fn = run_classifier.model_fn_builder(
      bert_config=modeling.BertConfig.from_json_file(CONFIG_FILE),
      num_labels=len(label_list),
      init_checkpoint=INIT_CHECKPOINT,
      learning_rate=LEARNING_RATE,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=False,
      use_one_hot_embeddings=True
    )
    estimator_from_checkpoints = tf.contrib.tpu.TPUEstimator(
      use_tpu=False,
      model_fn=model_fn,
      config=get_run_config(OUTPUT_DIR),
      train_batch_size=TRAIN_BATCH_SIZE,
      eval_batch_size=EVAL_BATCH_SIZE,
      predict_batch_size=PREDICT_BATCH_SIZE,
    )


def predict(sentence1, sentence2):
    start = time.time()
    global model_fn, estimator_from_checkpoints
    if model_fn == None:
        init()
    print('~~~~Model loaded after: {}'.format((time.time())-start))
    prediction_examples = []
    prediction_examples.append(run_classifier.InputExample('guid - 409', text_a = sentence1, text_b=sentence2, label="1"))
    input_features = run_classifier.convert_examples_to_features(prediction_examples,
                                                             label_list, MAX_SEQ_LENGTH, tokenizer)
    print('~~~~Features extracted after: {}'.format((time.time())-start))
    predict_input_fn = run_classifier.input_fn_builder(features=input_features,
                                                       seq_length=MAX_SEQ_LENGTH, is_training=False,
                                                       drop_remainder=True)
    predictions = estimator_from_checkpoints.predict(predict_input_fn)
    similarity = 0
    for example, prediction in zip(prediction_examples, predictions):
        print('~~~~text_a: %s\ntext_b: %s\nlabel:%s\nprediction:%s\n' % (example.text_a, example.text_b, str(example.label), prediction['probabilities']))
        similarity = float(prediction['probabilities'][1])
        print('~~~~Prediction inferenced after: {}'.format((time.time())-start))
    return {'sentence1': sentence1, 'sentence2': sentence2, 'similarity': similarity}

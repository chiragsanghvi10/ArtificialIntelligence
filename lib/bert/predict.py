BERT_PRETRAINED_DIR='/home/absin/Documents/dev/bert/model/msrc/'
CONFIG_FILE = os.path.join(BERT_PRETRAINED_DIR, 'bert_config.json')
INIT_CHECKPOINT = os.path.join(BERT_PRETRAINED_DIR, 'bert_model.ckpt')
OUTPUT_DIR=BERT_PRETRAINED_DIR
SAVE_CHECKPOINTS_STEPS=99999999
ITERATIONS_PER_LOOP=1000
NUM_TPU_CORES=8
LEARNING_RATE=2e-5
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

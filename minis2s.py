"""Minimum sequence-to-sequence model to demonstrate how to use TensorFlow API.

Visit https://github.com/kiidax/minis2s for the original copy.

This implements basic sequence-to-sequence model with LSTM encoder and decoder to
translate an English word into phone symbols using CMU pronuncing dictionary
(http://www.speech.cs.cmu.edu/cgi-bin/cmudict). The model is very small and will run on
a notebook with 4GB memory.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random

import tensorflow as tf
from tensorflow.python.ops import lookup_ops

tf.logging.set_verbosity(tf.logging.INFO)

# Flag definitions

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_enum("mode", None, ["datagen", "train", "eval", "infer"], "Run mode.")
flags.DEFINE_string("data_dir", None, "Data directory.")
flags.DEFINE_string("output_dir", None, "Base output directory for run.")
flags.mark_flags_as_required(['mode', 'data_dir'])

# Parameters

def create_parameters():
  hparams = tf.contrib.training.HParams(
    batch_size=40,
    num_epochs=4,
    learning_rate=0.1,
    sos='<s>',
    eos='</s>',
    embed_size=32,
    num_units=64,
    beam_width=10,
    max_tgt_len=20,
    src_train_file=os.path.join(FLAGS.data_dir, 'train.src'),
    tgt_train_file=os.path.join(FLAGS.data_dir, 'train.tgt'),
    src_dev_file=os.path.join(FLAGS.data_dir, 'dev.src'),
    tgt_dev_file=os.path.join(FLAGS.data_dir, 'dev.tgt'),
    src_test_file=os.path.join(FLAGS.data_dir, 'test.src'),
    tgt_test_file=os.path.join(FLAGS.data_dir, 'test.tgt'),
    src_vocab_file=os.path.join(FLAGS.data_dir, 'vocab.src'),
    tgt_vocab_file=os.path.join(FLAGS.data_dir, 'vocab.tgt'),
    output_dir=FLAGS.output_dir)
  return hparams

# Data preparation

def datagen(hparams):
  """Split original cmudict files into train, dev and test set,
  and prepare vocab files.
  """
  cmudict_file = os.path.join(FLAGS.data_dir, 'cmudict-0.7b')
  cmuvocab_file = os.path.join(FLAGS.data_dir, 'cmudict-0.7b.symbols')

  for source_file in (cmudict_file, cmuvocab_file):
    if not os.path.exists(source_file):
      print('%s not found.' % source_file)
      print('Please visit http://www.speech.cs.cmu.edu/cgi-bin/cmudict and download.')
      return
  
  print("Reading %s..." % cmudict_file)
  with open(cmudict_file, 'r') as f:
    lines = [line.strip() for line in f]
  samples = [line.split() for line in lines if len(line) > 0 and not line.startswith(';')]
  samples = [(list(s[0]), s[1:]) for s in samples]
  samples = [(' '.join(src), ' '.join(tgt)) for src, tgt in samples]

  random.shuffle(samples)
  num_samples = len(lines)
  num_dev_samples = num_samples // 10
  num_test_samples = num_samples // 10
  num_train_samples = num_samples - num_dev_samples - num_test_samples

  write_samples(samples[:num_train_samples], "train")
  write_samples(samples[num_train_samples:-num_test_samples], "dev")
  write_samples(samples[-num_test_samples:], "test")

  print("Reading %s..." % cmuvocab_file)
  with open(cmuvocab_file, 'r') as f:
    vocab = [line.strip() for line in f]
  vocab = ['<unk>', '<s>', '</s>'] + vocab

  output_file = os.path.join(FLAGS.data_dir, 'vocab.tgt')
  print("Writing %s" % output_file)
  with open(output_file, 'w') as f:
    f.writelines([sym + '\n' for sym in vocab])

  vocab = list(set(tok for src, _ in samples for tok in src.split(' ')))
  vocab = sorted(vocab)
  vocab = ['<unk>', '<s>', '</s>'] + vocab

  output_file = os.path.join(FLAGS.data_dir, 'vocab.src')
  print("Writing %s" % output_file)
  with open(output_file, 'w') as f:
    f.writelines([sym + '\n' for sym in vocab])

  print('Done')

def write_samples(samples, output_prefix):
  output_file = os.path.join(FLAGS.data_dir, '%s.src' % output_prefix)
  print("Writing %s..." % output_file)
  with open(output_file, 'w') as f:
    f.writelines([src + '\n' for src, _ in samples])
  output_file = os.path.join(FLAGS.data_dir, '%s.tgt' % output_prefix)
  print("Writing %s..." % output_file)
  with open(output_file, 'w') as f:
    f.writelines([tgt + '\n' for _, tgt in samples])


# Vocabulary

def check_vocab(vocab_file):
  with open(vocab_file, 'r') as f:
    idx2sym = [line.strip() for line in f]
  return idx2sym, len(idx2sym)

def create_vocab_tables(hparams):
  UNK_ID = 0
  src_vocab_table = lookup_ops.index_table_from_file(hparams.src_vocab_file, default_value=UNK_ID)
  tgt_vocab_table = lookup_ops.index_table_from_file(hparams.tgt_vocab_file, default_value=UNK_ID)
  return src_vocab_table, tgt_vocab_table

# Dataset

def get_input(is_train, hparams):

  output_buffer_size = 100 * hparams.batch_size
  num_parallel_calls = 2

  sos = hparams.sos
  eos = hparams.eos
  src_vocab_table, tgt_vocab_table = create_vocab_tables(hparams)
  src_eos_id = 2#tf.cast(src_vocab_table.lookup(tf.constant(eos)), tf.int32)
  tgt_sos_id = 1#tf.cast(tgt_vocab_table.lookup(tf.constant(sos)), tf.int32)
  tgt_eos_id = 2#tf.cast(tgt_vocab_table.lookup(tf.constant(eos)), tf.int32)

  if is_train:
    src_dataset = tf.data.TextLineDataset(hparams.src_train_file)
    tgt_dataset = tf.data.TextLineDataset(hparams.tgt_train_file)
  else:
    src_dataset = tf.data.TextLineDataset(hparams.src_dev_file)
    tgt_dataset = tf.data.TextLineDataset(hparams.tgt_dev_file)
  dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))
  if is_train:
    dataset = dataset.repeat(hparams.num_epochs)
    dataset = dataset.shuffle(output_buffer_size, reshuffle_each_iteration=True)

  dataset = dataset.map(
    lambda src, tgt: (
      tf.string_split([src]).values,
      tf.string_split([tgt]).values),
    num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

  dataset = dataset.map(lambda src, tgt: (
    tf.cast(src_vocab_table.lookup(src), tf.int32),
    tf.cast(tgt_vocab_table.lookup(tgt), tf.int32)
  ), num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

  dataset = dataset.map(lambda src, tgt: (
    src,
    tf.concat([[tgt_sos_id], tgt], 0),
    tf.concat([tgt, [tgt_eos_id]], 0)
  ), num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

  dataset = dataset.map(
    lambda src, tgt_in, tgt_out: (
      src, tgt_in, tgt_out, tf.size(src), tf.size(tgt_in)),
    num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
    
  dataset = dataset.padded_batch(
    hparams.batch_size,
    padded_shapes=(
      tf.TensorShape([None]),  # src
      tf.TensorShape([None]),  # tgt_input
      tf.TensorShape([None]),  # tgt_output
      tf.TensorShape([]),  # src_len
      tf.TensorShape([])),  # tgt_len
    padding_values=(
      src_eos_id,  # src
      tgt_eos_id,  # tgt_input
      tgt_eos_id,  # tgt_output
      0,  # src_len -- unused
      0))  # tgt_len -- unused
      
  iterator = dataset.make_initializable_iterator()
  tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)
  src, tgt_in, tgt_out, src_len, tgt_len = iterator.get_next()
  return ({
    'source': src,
    'source_length': src_len
  }, {
    'target_input': tgt_in,
    'target_output': tgt_out,
    'target_length': tgt_len
  })

def create_embeddings(src_vocab_size,
                      tgt_vocab_size,
                      src_embed_size,
                      tgt_embed_size):
  with tf.variable_scope("embeddings"):
    with tf.variable_scope("encoder"):
      embedding_encoder = tf.get_variable(
        'embedding_encoder',
        (src_vocab_size, src_embed_size),
        tf.float32)
    with tf.variable_scope("decoder"):
      embedding_decoder = tf.get_variable(
        'embedding_decoder',
        (tgt_vocab_size, tgt_embed_size),
        tf.float32)
  return embedding_encoder, embedding_decoder

def build_encoder(src,
                  src_len,
                  embedding_encoder,
                  hparams):
  src = tf.transpose(src)

  with tf.variable_scope("encoder") as encoder_scope:
    encoder_emb_inp = tf.nn.embedding_lookup(
      embedding_encoder, src)

    cell = tf.nn.rnn_cell.LSTMCell(
      hparams.num_units,
      forget_bias=1.0)
    _, encoder_state = tf.nn.dynamic_rnn(
      cell,
      encoder_emb_inp,
      dtype=tf.float32,
      sequence_length=src_len,
      time_major=True,
      swap_memory=True,
      scope=encoder_scope)

  return encoder_state

def build_decoder(decoder_initial_state,
                  tgt_in,
                  tgt_out,
                  tgt_len,
                  embedding_decoder,
                  tgt_vocab_size,
                  mode,
                  hparams):
  tgt_sos_id = 1
  tgt_eos_id = 2

  with tf.variable_scope("decoder") as decoder_scope:
    cell = tf.nn.rnn_cell.LSTMCell(hparams.num_units, forget_bias=1.0)

    output_layer = tf.layers.Dense(
      tgt_vocab_size, use_bias=False, name="output_projection")

    if mode != tf.estimator.ModeKeys.PREDICT:
      tgt_in = tf.transpose(tgt_in)
      tgt_out = tf.transpose(tgt_out)

      decoder_emb_inp = tf.nn.embedding_lookup(
        embedding_decoder, tgt_in)

      # Helper
      helper = tf.contrib.seq2seq.TrainingHelper(
        decoder_emb_inp, tgt_len,
        time_major=True)

      # Decoder
      my_decoder = tf.contrib.seq2seq.BasicDecoder(
        cell,
        helper,
        decoder_initial_state)

      # Dynamic decoding
      outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
        my_decoder,
        output_time_major=True,
        swap_memory=True,
        scope=decoder_scope)

      logits = output_layer(outputs.rnn_output)

      crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=tgt_out, logits=logits)
      max_time = tf.shape(tgt_in)[0]
      target_weights = tf.sequence_mask(
        tgt_len, max_time, dtype=tf.float32)

      target_weights = tf.transpose(target_weights)

      loss = tf.reduce_sum(
        crossent * target_weights) / tf.to_float(hparams.batch_size)
      sample_id = tf.no_op()

    else: # tf.estimator.ModeKeys.PREDICT
      decoder_initial_state = tf.contrib.seq2seq.tile_batch(
        decoder_initial_state, multiplier=hparams.beam_width)

      start_tokens = tf.fill([hparams.batch_size], tgt_sos_id)
      end_token = tgt_eos_id
      beam_width = hparams.beam_width

      my_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
        cell=cell,
        embedding=embedding_decoder,
        start_tokens=start_tokens,
        end_token=end_token,
        initial_state=decoder_initial_state,
        beam_width=beam_width,
        output_layer=output_layer)

      # Dynamic decoding
      outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
        my_decoder,
        maximum_iterations=hparams.max_tgt_len,
        output_time_major=True,
        swap_memory=True,
        scope=decoder_scope)

      loss = tf.no_op()
      sample_id = outputs.predicted_ids

    return loss, sample_id

def model_fn(features, labels, mode, params):
  hparams = params['hparams']
  src = features['source']
  src_len = features['source_length']
  if labels:
    tgt_in = labels['target_input']
    tgt_out = labels['target_output']
    tgt_len = labels['target_length']
  else:
    tgt_in, tgt_out, tgt_len = None, None, None

  _, src_vocab_size = check_vocab(hparams.src_vocab_file)
  _, tgt_vocab_size = check_vocab(hparams.tgt_vocab_file)

  embedding_encoder, embedding_decoder = create_embeddings(
    src_vocab_size,
    tgt_vocab_size,
    hparams.embed_size,
    hparams.embed_size)
  with tf.variable_scope("dynamic_seq2seq"):
    encoder_state = build_encoder(
      src, src_len, embedding_encoder, hparams)
    loss, sample_id = build_decoder(
      encoder_state, tgt_in, tgt_out, tgt_len, 
      embedding_decoder, tgt_vocab_size, mode, hparams)

  if mode == tf.estimator.ModeKeys.PREDICT:
    sample_id = tf.transpose(sample_id)
    predictions = {
      'sample_id': sample_id
    }
    return tf.estimator.EstimatorSpec(mode, predictions=predictions)

  predict_count = tf.reduce_sum(tgt_len)
  ppl = tf.exp(loss * tf.to_float(hparams.batch_size) / tf.to_float(predict_count))

  if mode == tf.estimator.ModeKeys.EVAL:
    metrics = {
    }
    return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

  assert mode == tf.estimator.ModeKeys.TRAIN

  optimizer = tf.train.GradientDescentOptimizer(learning_rate=hparams.learning_rate)
  train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
  tf.summary.scalar('train_ppl', ppl)

  for v in tf.global_variables():
    print(v.name, v.shape)

  return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

def train(hparams):
    estimator = tf.estimator.Estimator(
      model_fn=model_fn,
      params={'hparams': hparams}, 
      model_dir=hparams.output_dir)
    estimator.train(
      input_fn=lambda: get_input(is_train=True, hparams=hparams))

def eval(hparams):
  pass

def infer(hparams):
  tgt_idx2sym, _ = check_vocab(hparams.tgt_vocab_file)
  estimator = tf.estimator.Estimator(
    model_fn=model_fn,
    params={'hparams': hparams},
    model_dir=hparams.output_dir)
  predictions = estimator.predict(
    input_fn=lambda: get_input(is_train=False, hparams=hparams))

  tgt_eos_id = 2
  for batch_prediction in predictions:
    for sample in batch_prediction['sample_id']:
      print(' '.join(tgt_idx2sym[tok] for tok in sample if tok != tgt_eos_id))

def main(unused):
  hparams = create_parameters()
  if FLAGS.mode == 'datagen':
    datagen(hparams)
  elif FLAGS.mode == 'train':
    train(hparams)
  elif FLAGS.mode == 'eval':
    eval(hparams)
  elif FLAGS.mode == 'infer':
    infer(hparams)
  else:
    raise ValueError("Unknown mode")

if __name__ == "__main__":
  tf.app.run(main)
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
from tensorflow.python.ops import lookup_ops

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("data_dir", "", "Data directory.")
flags.DEFINE_string("output_dir", "", "Base output directory for run.")
flags.DEFINE_string("mode", "", "run mode.")

def get_src_vocab():
    idx2sym = (
        [str(i) for i in range(0, 10)] +
        [chr(i) for i in range(ord('A'), ord('Z') + 1)] +
        list('!()-./?+\'"#%&')
    )
    idx2sym = ['<unk>', '<s>', '</s>'] + idx2sym
    return idx2sym

def get_tgt_vocab(file):
    with open(file, 'r', encoding='utf-8') as f:
        idx2sym = [x.strip() for x in f]
    idx2sym = ['<unk>', '<s>', '</s>'] + idx2sym
    return idx2sym

def get_vocab_table(hparams):
    UNK_ID = 0
    src_idx2sym = get_src_vocab()
    tgt_idx2sym = get_tgt_vocab(hparams.tgt_vocab_file)
    src_vocab_table = lookup_ops.index_table_from_tensor(tf.convert_to_tensor(src_idx2sym, tf.string), default_value=UNK_ID)
    tgt_vocab_table = lookup_ops.index_table_from_tensor(tf.convert_to_tensor(tgt_idx2sym, tf.string), default_value=UNK_ID)
    return src_vocab_table, tgt_vocab_table

def get_input(hparams):

    output_buffer_size = 100 * hparams.batch_size
    num_parallel_calls = 2

    src_vocab_table, tgt_vocab_table = get_vocab_table(hparams)
    src_eos_id = 2
    tgt_bos_id = 1
    tgt_eos_id = 2

    dataset = tf.data.TextLineDataset(hparams.input_file)
    dataset = dataset.repeat(10)
    #dataset = dataset.shuffle(10 * output_buffer_size)

    def f(x):
        x = x.strip()
        if len(x) == 0 or x.startswith(b';'):
            return False, '', ''
        else:
            x = x.split()
            if len(x) < 1:
                return False, '', ''
            return True, [chr(y) for y in x[0]], x[1:]

    dataset = dataset.map(lambda x: tf.py_func(f, [x], (tf.bool, tf.string, tf.string)),
        num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
    dataset = dataset.filter(lambda has_data, src, tgt: has_data,
        ).prefetch(output_buffer_size)
    dataset = dataset.map(lambda has_data, src, tgt: (src, tgt),
        num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

    dataset = dataset.map(lambda src, tgt: (
        tf.cast(src_vocab_table.lookup(src), tf.int32),
        tf.cast(tgt_vocab_table.lookup(tgt), tf.int32)
    ), num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

    dataset = dataset.map(lambda src, tgt: (
        src,
        tf.concat([[tgt_bos_id], tgt], 0),
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

def build_encoder(src, src_len, hparams):
    src = tf.transpose(src)

    embedding_encoder = tf.get_variable('embedding_encoder', (hparams.src_vocab_size, hparams.num_units), tf.float32)
    encoder_emb_inp = tf.nn.embedding_lookup(
        embedding_encoder, src)

    cell = tf.nn.rnn_cell.LSTMCell(hparams.num_units, forget_bias=1.0)
    _, encoder_state = tf.nn.dynamic_rnn(
        cell,
        encoder_emb_inp,
        dtype=tf.float32,
        sequence_length=src_len,
        time_major=True,
        swap_memory=True)
    return encoder_state

def build_decoder(decoder_initial_state, tgt_in, tgt_out, tgt_len, mode, hparams):
    tgt_in = tf.transpose(tgt_in)
    tgt_out = tf.transpose(tgt_out)

    embedding_decoder = tf.get_variable('embedding_decoder', (hparams.tgt_vocab_size, hparams.num_units), tf.float32)
    decoder_emb_inp = tf.nn.embedding_lookup(
        embedding_decoder, tgt_in)
    decoder_emb_inp = tf.Print(decoder_emb_inp, [tf.shape(decoder_emb_inp)])

    cell = tf.nn.rnn_cell.LSTMCell(hparams.num_units, forget_bias=1.0)

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
    outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
        my_decoder,
        output_time_major=True,
        swap_memory=True,
        scope='decoder')

    logits = tf.layers.dense(outputs.rnn_output, hparams.tgt_vocab_size)
    print('logits', logits.shape)
    logits = tf.Print(logits, [tf.shape(logits)])

    if mode == tf.estimator.ModeKeys.TRAIN:
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tgt_out, logits=logits)
        max_time = tf.shape(tgt_in)[0]
        target_weights = tf.sequence_mask(
            tgt_len, max_time, dtype=tf.float32)

        target_weights = tf.transpose(target_weights)
        target_weights = tf.Print(target_weights, [target_weights])

        loss = tf.reduce_sum(
            crossent * target_weights) / tf.to_float(hparams.batch_size)

        return loss

    assert mode == tf.estimator.ModeKeys.PREDICT

    return logits

def dump_input(hparams):
    src_idx2sym = get_src_vocab()
    tgt_idx2sym = get_tgt_vocab(hparams.tgt_vocab_file)
    graph = tf.Graph()
    with tf.Session(graph=graph) as sess:
        dataset = get_input(hparams)

        iterator = dataset.make_initializable_iterator()
        initializer = tf.random_uniform_initializer(-1, 1)
        tf.get_variable_scope().set_initializer(initializer)
        sess.run(tf.tables_initializer())
        sess.run(tf.global_variables_initializer())
        sess.run(iterator.initializer)
        src, tgt_in, tgt_out, src_len, tgt_len = iterator.get_next()
        for i in range(100):
            src_val, tgt_in_val, tgt_out_val = sess.run((src, tgt_in, tgt_out))
            for j in range(src_val.shape[0]):
                src_val2 = ''.join(src_idx2sym[x] for x in src_val[j])
                tgt_in_val2 = ' '.join(tgt_idx2sym[x] for x in tgt_in_val[j])
                tgt_out_val2 = ' '.join(tgt_idx2sym[x] for x in tgt_out_val[j])
            print(src_val2, tgt_in_val2, tgt_out_val2)

def model_fn(features, labels, mode, params):
    hparams = params['hparams']
    src = features['source']
    src_len = features['source_length']
    if labels is None:
        src_vocab_table, tgt_vocab_table = get_vocab_table(hparams)
        src = src_vocab_table.lookup(tf.convert_to_tensor(['S I M P L E'.split()], tf.string))
        src_len = tf.convert_to_tensor([6], tf.int32)
        tgt_in = tgt_vocab_table.lookup(tf.convert_to_tensor(['<s> S IH1 M P AH0 L'.split()], tf.string))
        tgt_out = tgt_vocab_table.lookup(tf.convert_to_tensor(['S IH1 M P AH0 L </s>'.split()], tf.string))
        tgt_len = tf.convert_to_tensor([7], tf.int32)
    else:
        tgt_in = labels['target_input']
        tgt_out = labels['target_output']
        tgt_len = labels['target_length']

    encoder_state = build_encoder(src, src_len, hparams)
    loss = build_decoder(encoder_state, tgt_in, tgt_out, tgt_len, mode, hparams)

    if mode == tf.estimator.ModeKeys.PREDICT:
        logits = loss
        predicted_classes = tf.argmax(logits, axis=2)
        predicted_classes = tf.transpose(predicted_classes)
        logits = tf.transpose(logits)
        predictions = {
            'class_ids': predicted_classes,
            #'probabilities': tf.nn.softmax(logits),
            #'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)

    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

def infer(hparams):
    tgt_idx2sym = get_tgt_vocab(hparams.tgt_vocab_file)
    estimator = tf.estimator.Estimator(model_fn=model_fn, params={'hparams': hparams}, model_dir=hparams.output_dir)
    predictions = estimator.predict(input_fn=lambda: get_input(hparams))

    for p in predictions:
        class_ids = p['class_ids']
        print(class_ids)
        print(' '.join(tgt_idx2sym[y] for y in class_ids))
        break

def main(unused):

    hparams = tf.contrib.training.HParams(
        batch_size=1,
        src_vocab_size=64,
        tgt_vocab_size=87,
        num_units=64,
        input_file=os.path.join(FLAGS.data_dir, 'cmudict-0.7b'),
        tgt_vocab_file=os.path.join(FLAGS.data_dir, 'cmudict-0.7b.symbols'),
        output_dir=FLAGS.output_dir
    )

    if FLAGS.mode == 'train':
        estimator = tf.estimator.Estimator(model_fn=model_fn, params={'hparams': hparams}, model_dir=hparams.output_dir)
        estimator.train(input_fn=lambda: get_input(hparams))
    elif FLAGS.mode == 'infer':
        infer(hparams)
    else:
        dump_input(hparams)

if __name__ == "__main__":
    tf.app.run(main)
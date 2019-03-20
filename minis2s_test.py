
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
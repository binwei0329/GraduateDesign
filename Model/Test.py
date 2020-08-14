# Seq2Seq
# def build_encoder(encoder_input, vocab_size, embedding_dim, units):
#     # shape: (batch_size, inputs.shape[1], embedding_dim)
#     encoder_embed = Embedding(vocab_size, embedding_dim)(encoder_input)
#     # shape: (batch_size, inputs.shape[1], encoder_units), (batch_size, encoder_units)
#     # output, encoder_hid, encoder_cell = LSTM(units, return_sequences=True, return_state=True, dropout=0.4,
#     #                                          recurrent_initializer='glorot_uniform')(encoder_embed)
#
#     forward_lstm = LSTM(units, return_sequences=True, go_backwards=False, dropout=0.4,
#                         recurrent_initializer="glorot_uniform", name="forward_lstm")
#     backward_lstm = LSTM(units, return_sequences=True, go_backwards=True, dropout=0.4,
#                          recurrent_initializer="glorot_uniform", return_state=True, name="backward_lstm")
#
#     bilstm = Bidirectional(merge_mode="concat", layer=forward_lstm, backward_layer=backward_lstm, name="bilstm")
#     model = Model(encoder_input, [output, encoder_hid])
#
#     return model

# def decoder_model(decoder_input, hidden, encoder_outputs, vocab_size, embedding_dim, units):
#     # shape (batch_size, 1, embedding_dim) decoder
#     decoder_embed = Embedding(vocab_size, embedding_dim)(decoder_input)
#
#     # shape (batch_size, encoder_units), (batch_size, encoder_seq_len, 1)
#     context_vector, attention_weights = BahdanauAttention(units, trainable=True)([hidden, encoder_outputs])
#
#     # shape (batch_size, 1, embedding_dim + encoder_units)
#     output = Concatenate(axis=-1)([tf.expand_dims(context_vector, 1), decoder_embed])
#
#     # shape (batch_size, 1, decoder_units), (batch_size, decoder_units)
#     output, decoder_hid, decoder_cell = LSTM(units, return_sequences=True, return_state=True, dropout=0.2,
#                          recurrent_initializer='glorot_uniform')(output)
#     # (batch_size * 1, decoder_units)
#     output = Reshape(target_shape=(-1, units))(output)
#     # shape (batch_size, vocab_size)
#     output = Dense(vocab_size)(output)
#     model = Model(inputs=[decoder_input, hidden, encoder_outputs], outputs=[output, decoder_hid])
#
#     return model
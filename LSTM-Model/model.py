from keras.layers import LSTM, Dense, Embedding, Bidirectional
from keras.models import Sequential
from keras.optimizers import Adam

class SentimentLSTM():

    def __init__(self, n_classes, vocab_size, max_len, num_units=128,
                 useBiDirection=False, useAttention=False, learning_rate=0.001, dropout=0, embedding_size=300):
        self.model = Sequential()
        self.model.add(Embedding(input_dim=vocab_size,
                                 output_dim=embedding_size, input_length=max_len))
        lstm_model = LSTM(num_units, dropout=dropout)
        if useBiDirection:
            lstm_model = Bidirectional(lstm_model)
        if useAttention:
            lstm_model = lstm_model
            print("Attention not implement yet ... ")
        self.model.add(lstm_model)
        self.model.add(Dense(n_classes, activation='softmax'))

        self.model.summary()
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=Adam(lr=learning_rate),
                           metrics=['accuracy'])
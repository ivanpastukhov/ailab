from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Conv1D, Dense, Dropout, MaxPool1D, Concatenate, GlobalMaxPooling1D


def build_dssm(emb_mat, bias_initializer):
    embedding = Embedding(*emb_mat.shape, weights=emb_mat, trainable=False, mask_zero=True)
    conv_1 = Conv1D(128, kernel_size=3, strides=1, padding='valid', activation='relu')
    conv_2 = Conv1D(64, kernel_size=3, strides=1, padding='valid', activation='relu')


    question = Input((None, ), name='question')
    q = embedding(question)
    q = conv_1(q)
    q = MaxPool1D(3)(q)
    q = conv_2(q)
    q = GlobalMaxPooling1D(256)(q)

    answer = Input((None,), name='answer')
    a = embedding(answer)
    a = conv_1(a)
    a = MaxPool1D(3)(a)
    a = conv_2(a)
    a = GlobalMaxPooling1D(256)(a)

    x = Concatenate()[q,a]
    x = Dense(128, activation='relu')(x)
    out = Dense(1, activation='softmax', name='prediction', bias_initializer=bias_initializer)(x)
    return Model(inputs=(question, answer), outputs=out)





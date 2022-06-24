from preprocess.import_train_test_split import X_train, y_train
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense,  Concatenate
from tensorflow.keras.optimizers import Adam

# encoder katmanı
enc_input = Input(shape=(X_train.shape[1:]))
_, state_h, state_c = LSTM(256, return_state=True)(enc_input)
states = Concatenate(axis=-1)([state_h, state_c])
bottle_neck = Dense(128, activation='relu')(states)

# decoder katmanı
state_h_decoded = Dense(256, activation='relu')(bottle_neck)
state_c_decoded = Dense(256, activation='relu')(bottle_neck)
encoder_states = [state_h_decoded, state_c_decoded]
dec_input = Input(shape=(X_train.shape[1:]))
dec1 = LSTM(256, return_sequences=True)(dec_input, initial_state=encoder_states)
output = Dense(y_train.shape[2], activation='softmax')(dec1)

model = Model(inputs=[enc_input, dec_input], outputs=output) 
batch_size = 32
steps_per_epoch = len(X_train) // batch_size
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['acc'])

encoder_model = Model(inputs=model.layers[0].input, outputs=model.layers[3].output)

latent_input = Input(shape=(128, ))
state_h = model.layers[5](latent_input)
state_c = model.layers[6](latent_input)
latent_to_states_model = Model(latent_input, [state_h, state_c])

decoder_inputs = Input(batch_shape=(1, 1, 54))
decoder_lstm = LSTM(256, return_sequences=True, stateful=True)(decoder_inputs)
decoder_outputs = Dense(54, activation='softmax')(decoder_lstm)
gen_model = Model(decoder_inputs, decoder_outputs)
for i in range(1,3):
    gen_model.layers[i].set_weights(model.layers[i+6].get_weights())

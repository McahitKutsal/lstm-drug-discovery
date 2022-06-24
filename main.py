from preprocess.generator import Data_Generator
from preprocess.import_train_test_split import X_train, y_train, X_val, y_val
from utils.network import model
from utils.network import encoder_model
from utils.sample_from_latent import generate

batch_size = 32
training_generator = Data_Generator(X_train, y_train, batch_size)
validation_generator = Data_Generator(X_val, y_val, batch_size)

nb_epochs = 100
validation_steps = len(X_val) // batch_size
steps_per_epoch = len(X_train) // batch_size
history = model.fit(training_generator, steps_per_epoch=steps_per_epoch, epochs=nb_epochs, verbose=1, 
                              validation_data=validation_generator, validation_steps=validation_steps, 
                             use_multiprocessing=False, shuffle=True, callbacks=[])

latent_space = encoder_model.predict(X_train)
latent_seed = latent_space[50:51]
sampling_temp = 0.75
scale = 0.5
quantity = 150
t_mols, t_smiles = generate(latent_seed, sampling_temp, scale, 60)
print(t_smiles)


import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from src.modeling import correlation_gan as cGAN  # Assuming you have a correlation_gan module
from src.preparation import data_preproc as preproc


# Set GPU memory allocation
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

# Load and preprocess data
X_train, scaling = preproc.dataExtraction_puma(
    DB_path='./data/raw/data_correlation_gan.h5',  # Adjust the path and file name
    DB_name='dataset', im_shape=(64, 128, 81)
)

# Create Correlation GAN model
correlation_gan = cGAN.CorrelationGAN(latent_dim=64, target_shape=(64, 128, 82), batch_size=2,
                                    optimizerG=None, optimizerC=None, summary=True,
                                    n_critic=1, models=None, gradient_penalty=10,
                                    data=X_train[:100, :, :, :], tfboard=False)

# Train the Correlation GAN
correlation_gan.train(epochs=5, save_interval=4, save_file='correlation_gan_model',
                      run_name='run_correlation_gan', log_interval=10,
                      log_file='correlation_gan_logs', data_generator=None,
                      save_intermediate_model=True)

#!/usr/bin/env python
# coding: utf-8

from db_utils import *
from tensorflow.keras.models import load_model
from wgan import *

parser = argparse.ArgumentParser()
parser.add_argument('run', type=int)
parser.add_argument('number', type=int)
args = parser.parse_args()

run = args.run
number = args.number

path_gen = f'runs/{run}/{number}_gen.h5'
path_critic = f'runs/{run}/{number}_critic.h5'

print(path_gen)
print(path_critic)


def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)

tensorflow.keras.losses.wasserstein_loss = wasserstein_loss
gen = load_model(path_gen, custom_objects={'wasserstein_loss': wasserstein_loss})
critic = load_model(path_critic, custom_objects={'wasserstein_loss': wasserstein_loss})
noise_dim = 100
ncritic = 5
wgan = WGAN(gen, critic, noise_dim, ncritic)
db_train , db_test = load_data()
wgan.train(3000, db_train, db_test)

#trajs = wgan.gen.predict(np.random.normal(0, 1, size=(50000, noise_dim)))
#np.save(f'runs/{run}/gen_trajs', trajs)

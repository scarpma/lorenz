#!/usr/bin/env python
# coding: utf-8

from db_utils import *
from gen import *
from critic import *

class WGAN():
    def __init__(self,gen,critic,noise_dim,n_critic):
        self.d_losses = []
        self.g_losses = []
        self.real_critics = []
        self.fake_critics = []
        self.sig_len = 2000
        self.channels = 1
        self.noise_dim = noise_dim

        # Creo cartella della run attuale:
        self.current_run = self.get_run()
        self.dir_path = f"/scratch/scarpolini/lorenz_wgan/runs/{self.current_run}/"
        os.mkdir(self.dir_path)

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = n_critic
        self.clip_value = 0.2
        optimizer = RMSprop(lr=0.00005)

        # compile the critic
        self.critic = critic
        self.critic.compile(loss=self.wasserstein_loss, optimizer=optimizer)

        # generator
        self.gen = gen

        self.critic.trainable = False
        gan_input = Input(shape=(self.noise_dim,))
        fake_traj = self.gen(gan_input)
        gan_output = self.critic(fake_traj)
        self.gan = Model(gan_input, gan_output)
        self.gan.compile(loss=self.wasserstein_loss, optimizer=optimizer)
        
    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    
    def get_run(self):
        runs = glob.glob('runs/*/')
        runs = sorted(runs, key=lambda x: int(x[5:-1]))
        # print('RUNS = ',runs)
        runs = [int(line[5:-1]) for line in runs]

        return runs[-1] + 1

 
    def plot_trajs(self, gen_trajs,epoch):
        plt.figure(figsize=(15, 2))
        for i, traj in enumerate(gen_trajs):
            plt.subplot(1, 1, i+1)
            plt.plot(traj)
        plt.tight_layout()
        plt.savefig(self.dir_path+f'{epoch}_gen_traj.png', fmt='png', dpi=220)
        plt.close()

    def plot_disc_predictions(self, fake_disc, real_disc, epoch, batch_size=250):
        plt.figure()
        plt.plot(real_disc, label='real')
        plt.plot(fake_disc, label='fake')
        plt.legend()
        plt.savefig(self.dir_path+f'{epoch}_critic_pred.png', fmt='png', dpi=220)
        plt.close()
    
    def plot_training(self):
        fig, (ax1, ax2) = plt.subplots(1,2,figsize=(16,5))
        ax1.set_xlabel('epochs')
        ax2.set_xlabel('epochs')
        ax1.set_title('Losses')
        ax2.set_title('Critic predictions')
        ax1.plot(self.d_losses, label='critic loss')
        ax1.plot(self.g_losses, label='generator loss')
        fake_critic = [a[0] for a in self.fake_critics]
        fake_critice = [a[1] for a in self.fake_critics]
        real_critic = [a[0] for a in self.real_critics]
        real_critice = [a[1] for a in self.real_critics]
        n_epochs = len(fake_critic)
        l, caps, c = ax2.errorbar(range(n_epochs), fake_critic, fake_critice,lw=0,
                                  marker='^', ms=2, elinewidth=1, uplims=True,
                                  lolims=True, capsize=1, label='fake')
        for cap in caps:
            cap.set_marker("_")
        l, caps, c = ax2.errorbar(range(n_epochs), real_critic, real_critice, lw=0,
                                  marker='^', ms=2, elinewidth=1, uplims=True,
                                  lolims=True, capsize=1, label='real')
        for cap in caps:
            cap.set_marker("_")
            
        ax1.legend()
        ax2.legend()
        plt.savefig(self.dir_path+f'training.png', fmt='png', dpi=220)
        plt.close()

    def train(self, epochs, db_train, batch_size=250):
        
        # salvo info log #
        with open(self.dir_path+'logs.txt','a+') as f:
            f.write(f"ncritic={self.n_critic}\nepochs={epochs}")
        fl = open(self.dir_path+'training.dat','a+')
        # ############## #
        
        static_noise = np.random.normal(0, 1, size=(1, self.noise_dim))
        mini_batch_size = batch_size * self.n_critic
        len_data = len(db_train[:,0,0])
        steps_per_epoch = len_data // (mini_batch_size)
        valid = -np.ones(batch_size)
        fake  =  np.ones(batch_size)
        disc_labels = np.concatenate((valid, fake))
        
        print(f'\nNCRITIC = {self.n_critic}\n')
 
        for epoch in range(epochs):
            for ii in range(len_data // mini_batch_size):
                
                real_trajs = db_train[np.arange(ii*mini_batch_size, (ii+1)*mini_batch_size)]
                noise = np.random.normal(0, 1, size=(mini_batch_size, self.noise_dim))
                fake_trajs = self.gen.predict(noise)
                for jj in range(self.n_critic):
                    
                    trajs = np.concatenate((real_trajs[jj*batch_size:(jj+1)*batch_size],
                                            fake_trajs[jj*batch_size:(jj+1)*batch_size]))
                    d_loss = self.critic.train_on_batch(trajs, disc_labels)
                    
                    # Clip critic weights
                    for l in self.critic.layers:
                        weights = l.get_weights()
                        weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                        l.set_weights(weights)
                    
                noise = np.random.normal(0, 1, size=(batch_size, self.noise_dim))
                g_loss = self.gan.train_on_batch(noise, valid)
                print(f'{ii}/{len_data // mini_batch_size}', end='\r')
    
            n = round(np.random.uniform(0,len_data-batch_size))
            real_critic = self.critic.predict(db_train[n:n+batch_size])
            sample_gen = self.gen.predict(noise)
            fake_critic = self.critic.predict(sample_gen)
            if epoch%10==0:
                sample_gen = self.gen.predict(static_noise)
                self.plot_trajs(sample_gen, epoch)
                self.plot_disc_predictions(fake_critic, real_critic, epoch, batch_size)
            if epoch%50==0:
                self.critic.save(self.dir_path+f'{epoch}_disc.h5')
                self.gen.save(self.dir_path+f'{epoch}_gen.h5')
                
            real_critic = [np.mean(real_critic), np.std(real_critic)]
            fake_critic = [np.mean(fake_critic), np.std(fake_critic)]
            self.g_losses.append(g_loss)
            self.d_losses.append(d_loss)
            self.real_critics.append(real_critic)
            self.fake_critics.append(fake_critic)
            text = f"epoch: {epoch} [D loss: {d_loss:.4f}] [G loss: {g_loss:.4f}] [D accuracy: {(real_critic[0]-fake_critic[0])/np.max([real_critic[1],fake_critic[1]]):.4f}]"
            print (text)
            fl.write(text+'\n')
        
        fl.close()
        self.critic.save(self.dir_path+f'{epochs-1}_critic.h5')
        self.gen.save(self.dir_path+f'{epochs-1}_gen.h5')
        self.plot_training()
        

            
if __name__ == '__main__' :
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--ncritic', type=int, default=5)
    args = parser.parse_args()
    epochs = args.epochs
    ncritic = args.ncritic
    
    db_train , _ = load_data()
    
    noise_dim = 100
    gen = build_generator(fs=(20,1), fm=4, init_sigma=0.2, init_mean=0.03, alpha=0.3, noise_dim=noise_dim)
    critic = build_critic(fs=20, fm=4, init_sigma=0.2, init_mean=0.0)
    wgan = WGAN(gen, critic, noise_dim, ncritic)
    print(f'train for {epochs} epochs')
    wgan.train(epochs, db_train)

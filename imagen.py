from torch import nn
import torch
from matplotlib import pyplot as plt


GEN_PATH = 'weights/gen6.pth'
DISC_PATH = 'weights/disc6.pth'
NOISE_DIM = 100
    
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_dim+10, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.Dropout(0.55),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.Dropout(0.55),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.label_layer = nn.Sequential(
            nn.Embedding(10, 10)
        )
    
    def forward(self, x, labels):
        labels = self.label_layer(labels)
        x = torch.cat((x, labels), 1)
        return self.linear(x)
    
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_dim+10, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.001),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.001),
            nn.Linear(512, output_dim),
            nn.Tanh()
        )
        
        self.label_layer = nn.Sequential(
            nn.Embedding(10, 10)
        )
        
    def forward(self, x, labels):
        labels = self.label_layer(labels)
        x = torch.cat((x, labels), 1)
        return self.linear(x)
    
def num_to_list(n):
    arr = []
    while n != 0:
        arr.append(n % 10)
        n //= 10
    return arr[::-1]

def gen_num(n, g, gen):
    n = num_to_list(n)
    names = []
    for j in range(1, g+1):
        noise = torch.randn(1, NOISE_DIM)
        fig = plt.figure()
        for idx, i in enumerate(n):
            fig.add_subplot(1, len(n), idx+1)
            i = torch.LongTensor([i])
            plt.axis('off')
            prev_noise = noise
            noise += (torch.rand(NOISE_DIM)-0.5)*0.5
            plt.imshow(gen(noise, i)[0].reshape(28, 28).detach().numpy()[:, 3:-2], cmap='binary')
            noise = prev_noise
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig(f'static/images/im{j}.png')
        names.append(f'static/images/im{j}.png')
    plt.clf()
    plt.close()
    return names

def save_num(num, generations):
    disc = Discriminator(784)
    gen = Generator(NOISE_DIM, 784)
    disc.load_state_dict(torch.load(f=DISC_PATH))
    gen.load_state_dict(torch.load(f=GEN_PATH))
    return gen_num(num, generations, gen)

import random
import gym
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torchvision import transforms
from PIL import Image


class ReplayMemory:
    """Memory that store {s, a, r, s_}

    Attributes
    ----------
    capacity : int
        memory size
    s : torch.empty
        store state, shape: (capacity, *state_shape)
    a : torch.empty
        store action, shape: (capacity, 1)
    r : torch.empty
        store reward, shape: (capacity, 1)
    s_ : torch.empty
        store next state, shape: (capacity, *state_shape)
    save_idx : int
        index of memory to save data
    size_idx : int
        number of data in memory

    Method
    ------
    push(self, s, a, r, s_)
        save s, a, r, s_ into memory
    sample(self, batch_size)
        sample batch_size data from memory
    __len__(self)
        return number of data in memory
    """

    def __init__(self, capacity, state_shape):
        """
        Parameters
        ----------
        capacity : int
            memory size
        s : torch.empty
            store state, shape: (capacity, *state_shape)
        a : torch.empty
            store action, shape: (capacity, 1)
        r : torch.empty
            store reward, shape: (capacity, 1)
        s_ : torch.empty
            store next state, shape: (capacity, *state_shape)
        save_idx : int
            index of memory to save data
        size_idx : int
            number of data in memory
        """

        self.capacity = capacity
        self.s = torch.empty((self.capacity, *state_shape))
        self.a = torch.empty((self.capacity, 1))
        self.r = torch.empty((self.capacity, 1))
        self.s_ = torch.empty((self.capacity, *state_shape))
        self.save_idx = 0
        self.size_idx = 0

    def push(self, s, a, r, s_):
        """save s, a, r, s_ into memory
        
        Parameters
        ----------
        s : torch.tensor
            state
        a : torch.tensor
            action
        r : torch.tensor
            reward
        s_ : torch.tensor
            next state
        """

        self.s[self.save_idx] = s
        self.a[self.save_idx] = a
        self.r[self.save_idx] = r
        self.s_[self.save_idx] = s_

        self.save_idx = (self.save_idx+1)%self.capacity
        self.size_idx = min(self.size_idx+1, self.capacity)

    def sample(self, batch_size):
        """sample batch_size data from memory

        Parameters
        ----------
        batch_size : int
            number of data to sample from memory

        Returns
        -------
        (s, a, r, s_) : torch tensor in tuple
            batch_size of {s, a, r, s_} sample from memory
        """
        sample_idx = np.random.choice(self.capacity, batch_size)

        return (self.s[sample_idx],
                self.a[sample_idx],
                self.r[sample_idx],
                self.s_[sample_idx])

    def __len__(self):
        """return number of data in memory
        """

        return self.size_idx


class DQN(nn.Module):
    def __init__(self, h, w, outputs):
        """DQN

        Attributes
        ----------
        h : int
            image height
        w : int
            image width
        outputs : int
            size of output feature

        Methods
        -------
        forward(self, x)
            given x as NN input, return the NN output
        """

        super(DQN, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        """given x as NN input, return the NN output

        Parameters
        ----------
        x : torch.tensor
            model input
        """

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

def get_cart_location(screen_width):
    """return cart location in screen scale

        env.x_threshold
            max cart location in gym scale, i.e., 2.4
        screen scale
            -300 --- 0 --- 300
        gym scale
            -2.4 --- 0 --- 2.4

    Parameters
    ----------
    screen_width : int
        
    Returns
    -------
    cart_location_scaled : int
        cart location in screen scale
        origin point is at most left, not middle
    """

    cart_location = env.state[0]
    scale = (screen_width/2) / env.x_threshold
    cart_location_scaled = int(cart_location*scale+screen_width/2)

    return cart_location_scaled

def get_screen():
    screen = env.render(mode='rgb_array').transpose((2,0,1)) #HWC->CHW
    # screen.shape = (3, 400, 600)

    h, w = screen.shape[1:]
    # slice screen
    screen = screen[:, int(h*0.4):int(h*0.8)]
    view_width = int(w*0.6)

    ### origin point is at most left!!!!!

    cart_location = get_cart_location(w)
    # cart at left side
    if cart_location < view_width//2:
        slice_range = slice(view_width)
    # cart at right side
    elif cart_location > (w-view_width//2):
        slice_range = slice(-view_width, None)
    # other position
    else:
        slice_range = slice(cart_location-view_width//2,
                            cart_location+view_width//2)
    screen = screen[:,:,slice_range]
    screen = np.ascontiguousarray(screen, dtype=np.float32)/255
    screen = torch.from_numpy(screen)

    resize = transforms.Compose([transforms.ToPILImage(),
                                 transforms.Resize(40, interpolation=Image.CUBIC),
                                 transforms.ToTensor()])
    screen = resize(screen).unsqueeze(0).to(device)
    return screen

env = gym.make('CartPole-v0').unwrapped
device = torch.device('cuda:0')

env.reset()
BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

init_screen = get_screen()
_, _, h, w = init_screen.shape

n_action = env.action_space.n

policy_net = DQN(h, w, n_action).to(device)
target_net = DQN(h, w, n_action).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000, (h, w))

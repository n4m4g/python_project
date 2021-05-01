import numpy as np


class ReplayBuffer:

    def __init__(self, max_size, input_shape, n_actions):
        # define memory capacity, memory pointer
        self.mem_size = max_size
        self.mem_ptr = 0

        # define memory contents
        self.s_mem = np.zeros((max_size, input_shape))
        self.ns_mem = np.zeros((max_size, input_shape))
        self.a_mem = np.zeros((max_size, n_actions))
        self.r_mem = np.zeros(max_size)
        self.t_mem = np.zeros(max_size, dtype=np.bool)

    def store_transition(self, s, ns, a, r, done):
        # get memory idx in valid memory access range
        idx = self.mem_ptr % self.mem_size

        # save transition
        self.s_mem[idx] = s
        self.ns_mem[idx] = ns
        self.a_mem[idx] = a
        self.r_mem[idx] = r
        self.t_mem[idx] = done

        # increase the memory pointer after the store operation
        self.mem_ptr += 1

    def sample_buffer(self, batch_size):
        # get memory valid size
        max_mem = min(self.mem_ptr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)
        s = self.s_mem[batch]
        ns = self.ns_mem[batch]
        a = self.a_mem[batch]
        r = self.r_mem[batch]
        t = self.t_mem[batch]

        return s, ns, a, r, t

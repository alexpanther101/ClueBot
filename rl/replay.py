#replay buffer
import random
from collections import deque, namedtuple
import numpy as np

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done', 'mask'))

class ReplayBuffer:
    def __init__(self, capacity, n_step=3, gamma=0.99):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        
        # N-step support
        self.n_step = n_step
        self.gamma = gamma
        self.n_step_queue = deque(maxlen=n_step)

    def push(self, state, action, reward, next_state, done, mask):
        """
        Stores a single step, builds N-step transition when queue is ready.
        """
        self.n_step_queue.append((state, action, reward, next_state, done, mask))
        
        # Only add to buffer when we have n steps
        if len(self.n_step_queue) == self.n_step:
            self._commit_n_step()

        # If episode ends early, flush remaining queue
        if done:
            while self.n_step_queue:
                self._commit_n_step()

    def _commit_n_step(self):
        """
        Compute N-step return for first element in queue and store it.
        """
        state, action, _, _, _, mask = self.n_step_queue[0]
        reward, next_state, done = self._compute_n_step_return()

        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = Transition(state, action, reward, next_state, done, mask)
        self.pos = (self.pos + 1) % self.capacity

        self.n_step_queue.popleft()

    def _compute_n_step_return(self):
        """
        Computes cumulative reward, final next_state, and done flag.
        """
        R = 0.0
        for idx, (_, _, r, _, d, _) in enumerate(self.n_step_queue):
            R += (self.gamma ** idx) * r
            if d:  # Stop discounting at terminal state
                break
        next_state, done = self.n_step_queue[-1][3], self.n_step_queue[-1][4]
        return R, next_state, done

    def sample(self, batch_size):
        """
        Sample random batch for training.
        """
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)

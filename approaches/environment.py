import abc

class Environment:
        
    @abc.abstractmethod
    def randomize_rewards(self, rng):
        raise NotImplementedError('Override me!')
        # check reward fn somehow?

    @abc.abstractmethod
    def step(self, action):
        raise NotImplementedError('Override me!')

    @abc.abstractmethod
    def reset(self, rng):
        raise NotImplementedError('Override me!')
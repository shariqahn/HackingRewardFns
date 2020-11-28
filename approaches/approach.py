import abc

class Approach:
        
    @abc.abstractmethod
    def reset(self, reward_function):
        raise NotImplementedError('Override me!')

    @abc.abstractmethod
    def get_action(self, state):
        raise NotImplementedError('Override me!')

    @abc.abstractmethod
    def observe(self, state, action, next_state, reward, done):
        raise NotImplementedError('Override me!')
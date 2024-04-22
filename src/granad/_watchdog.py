# this is essentially a global variable, non-threadsafe etc
class _Watchdog:
    _counter = 0

    @classmethod
    def next_value(cls):
        cls._counter += 1
        return cls._counter

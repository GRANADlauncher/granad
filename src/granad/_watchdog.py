# this is essentially a global variable, non-threadsafe etc
from dataclasses import dataclass

@dataclass(frozen = True)
class GroupId():
    id : int = 0

    def __eq__(self, other):
        if not isinstance(other, GroupId):
            return NotImplemented
        return self.id == other.id

    def __lt__(self, other):
        if not isinstance(other, GroupId):
            return NotImplemented
        return self.id < self.id

    def __le__(self, other):
        return self < other or self == other

    def __gt__(self, other):
        return not self <= other

    def __ge__(self, other):
        return not self < other

    def __ne__(self, other):
        return not self == other

class _Watchdog:
    _counter = GroupId()

    @classmethod
    def next_value(cls):
        cls._counter = GroupId( cls._counter.id + 1)
        return cls._counter

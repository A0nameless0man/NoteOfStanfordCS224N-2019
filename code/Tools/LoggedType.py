class LoggedFloat:
    def __init__(self, min=float('+inf'), max=float('-inf')):
        self._MIN_ = min
        self._MAX_ = max
        self._sum_ = 0.0
        self._count_ = 0
        self.reset()

    def reset(self):
        self._min_ = self._MIN_
        self._max_ = self._MAX_
        self._sum_ = 0.0
        self._count_ = 0

    def update(self, value):
        self._sum_ += value
        self._count_ += 1
        self._min_ = min(self._min_, value)
        self._max_ = max(self._max_, value)
        return self

    @property
    def min(self):
        return self._min_

    @property
    def max(self):
        return self._max_

    @property
    def count(self):
        return self._count_

    @property
    def sum(self):
        return self._sum_

    # @property
    # def mean(self):
    #     return self._sum_ / self._count_

    @property
    def avg(self):
        return self._sum_ / self._count_

    @property
    def average(self):
        return self._sum_ / self._count_

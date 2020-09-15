class Counter:
    def __init__(self, initial=0):
        self._cnt_ = initial

    def reset(self):
        self._cnt_ = 0

    def lap(self):
        return Counter(self._cnt_)

    def step(self):
        self._cnt_ += 1
        return self

    def distance(self, other):
        return abs(self._cnt_ - other._cnt_)

    def mean(self, other):
        return abs(self._cnt_ + other._cnt_) // 2

    @property
    def value(self):
        return self._cnt_
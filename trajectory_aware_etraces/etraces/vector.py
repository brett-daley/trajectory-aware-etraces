import numpy as np


class Vector:
    """A dynamically sized NumPy array optimized for eligibility-trace operations."""

    def __init__(self, dtype=np.float64, init_length=1_000, growth_factor=2.0):
        self._dtype = dtype
        self._growth_factor = growth_factor
        self._array = self._new_array(init_length)
        self._size = 0

    def __len__(self):
        return self._size

    def numpy(self):
        return self._array[:self._size]

    def append(self, value):
        self._array[self._size] = value
        self._size += 1
        if self._size >= len(self._array):
            self._increase_max_size()

    def assign(self, value):
        self._array[:self._size] = value

    def multiply(self, value):
        self._array[:self._size] *= value

    def _new_array(self, size):
        return np.empty(size, dtype=self._dtype)

    def _increase_max_size(self):
        old_array = self._array
        old_size = len(old_array)
        new_size = int(old_size * self._growth_factor)
        assert new_size > old_size

        new_array = self._new_array(new_size)
        new_array[:old_size] = old_array
        self._array = new_array

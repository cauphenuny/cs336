from functools import total_ordering
import heapq
from typing import TypeVar, Generic, final, Protocol, Any


class SupportsLessThan(Protocol):
    def __lt__(self, value: Any, /) -> bool: ...


K = TypeVar("K")
V = TypeVar("V")
L = TypeVar("L", bound=SupportsLessThan)


@total_ordering
class Entry(Generic[L]):
    def __init__(self, data: L):
        self.raw = data

    def __lt__(self, other: "Entry[L]") -> bool:
        return other.raw < self.raw

    def __repr__(self) -> str:
        return repr(self.raw)

    def __str__(self) -> str:
        return str(self.raw)

    def __eq__(self, other: "Entry[L]") -> bool:
        return self.raw == other.raw


@final
class ReverseHeap(Generic[K, V]):
    """
    Counter, unique keys/values, and reverse order.
    """

    def __init__(self, init: list[tuple[K, V]] = [], reverse: bool = False):
        self.data: list[Entry[tuple[K, V]]] = [Entry((key, value)) for key, value in init]
        self.removed: list[Entry[tuple[K, V]]] = []
        heapq.heapify(self.data)

    def insert(self, key: K, value: V):
        heapq.heappush(self.data, Entry((key, value)))

    def remove(self, key: K, value: V):
        heapq.heappush(self.removed, Entry((key, value)))
        # print(f"removed: {key, value}, result: {self.removed}")

    def pop(self) -> tuple[K, V]:
        while self.data:
            top = heapq.heappop(self.data)
            # print(f"popped: {top.raw}, remaining: {self.data = }, {self.removed = }")
            if self.removed and top == self.removed[0]:
                # print(f"popped removed: {top.raw}, remaining removed: {self.removed}")
                heapq.heappop(self.removed)
                continue
            return top.raw
        raise IndexError("pop from an empty heap")

    def __getitem__(self, index: int) -> tuple[K, V]:
        return self.data[index].raw

    def __iter__(self):
        class Iterator:
            def __init__(self, heap: ReverseHeap[K, V]):
                self.heap = heap
                self.index = 0

            def __next__(self) -> tuple[K, V]:
                if self.index >= len(self.heap.data):
                    raise StopIteration
                item = self.heap.data[self.index]
                self.index += 1
                return item.raw

        return Iterator(self)

    def __len__(self) -> int:
        return len(self.data) - len(self.removed)

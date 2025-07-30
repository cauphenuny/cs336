import pytest
from ..heap import ReverseHeap


def test_reverse_heap_insert_and_pop():
    heap = ReverseHeap([(5, "a"), (3, "b"), (5, "c")])
    # Should pop the largest key first
    key, value = heap.pop()
    assert key == 5 and value == "c"
    key, value = heap.pop()
    assert key == 5 and value == "a"
    key, value = heap.pop()
    assert key == 3 and value == "b"
    with pytest.raises(IndexError):
        heap.pop()


def test_reverse_heap_insert():
    heap = ReverseHeap()
    heap.insert(2, "x")
    heap.insert(10, "y")
    heap.insert(5, "z")
    assert heap.pop() == (10, "y")
    assert heap.pop() == (5, "z")
    assert heap.pop() == (2, "x")


def test_reverse_heap_len():
    heap = ReverseHeap()
    assert len(heap) == 0
    heap.insert(1, "a")
    heap.insert(2, "b")
    assert len(heap) == 2
    heap.pop()
    assert len(heap) == 1
    heap.pop()
    assert len(heap) == 0


def test_reverse_heap_remove():
    heap = ReverseHeap()
    heap.insert(4, "a")
    heap.insert(7, "b")
    heap.insert(2, "c")
    heap.remove(7, "b")
    assert heap.pop() == (4, "a")
    assert heap.pop() == (2, "c")
    with pytest.raises(IndexError):
        heap.pop()


if __name__ == "__main__":
    test_reverse_heap_remove()

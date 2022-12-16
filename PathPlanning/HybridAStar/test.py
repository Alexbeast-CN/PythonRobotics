import heapq

heap = [10, 33, 13]

# Insert elements into the heap
heapq.heappush(heap, 4)
heapq.heappush(heap, 5)
heapq.heappush(heap, 3)
heapq.heappush(heap, 2)
heapq.heappush(heap, 1)

print(heap)

heapq.heappop(heap)

print(heap)
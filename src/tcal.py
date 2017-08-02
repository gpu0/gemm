"""
Calculate different thread patterns
"""

tx = list(range(0, 256))

print("gmLoadBtx ", map(lambda x: x%8 + (x/8)*16 + 1*256, tx))
print("ldsStoreBtx ", map(lambda x: (x%8)*128 + x/8 + 1*256, tx))
print("", map(lambda x: (x%2) * 128 * 4 + x, tx))
print(128*8/256)


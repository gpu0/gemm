"""
Calculate different thread patterns
"""

tx = list(range(0, 256))

print("gmLoadBtx ", map(lambda x: x%8 + (x/8)*16 + 1*256, tx))
print("ldsStoreBtx ", map(lambda x: (x%8)*128 + x/8 + 1*256, tx))
print("", map(lambda x: (x%2) * 512 + x/2, tx))
print("gmStoreCtx", map(lambda x: (x%16)*2 + (x/16)*8*128, tx))
print(128*8/256)


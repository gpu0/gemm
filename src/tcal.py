"""
Calculate different thread patterns
"""

tx = list(range(0, 256))

print("sAtx", map(lambda x: (x%2) * 512 + x/2, tx))
print("gmStoreCtx", map(lambda x: (x%16)*2 + (x/16)*8*128 + 7*32 + 1, tx))
print("a0", map(lambda x: x%16, tx))
print("a1", map(lambda x: x%16+16, tx))
print(128*128)

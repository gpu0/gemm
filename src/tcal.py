"""
Calculate different thread patterns
"""

tx = list(range(0, 256))

#print("sAtx", map(lambda x: (x%2) * 512 + x/2, tx))
print("gmStoreCtx", map(lambda x: (x%16)*2 + (x/16)*16*8, tx))
print("", map(lambda x: (x%16)*2 + (x/16)*16*8 + 32 + 1, tx))


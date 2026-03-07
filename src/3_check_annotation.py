import wfdb

annotation = wfdb.rdann('Data/07879', 'atr')

print("Symbols:")
print(annotation.symbol)

print("\nAux notes (REAL rhythm labels):")
print(annotation.aux_note)

print("\nSample positions:")
print(annotation.sample)
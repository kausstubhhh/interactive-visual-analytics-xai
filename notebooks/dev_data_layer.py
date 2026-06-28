from src.data.loader import load_heloc
from src.data.loader import load_bank

from src.data.validator import dataset_summary

print("\nHELOC DATASET")
print("=" * 50)

heloc = load_heloc()
print(dataset_summary(heloc))


print("\nBANK DATASET")
print("=" * 50)

bank = load_bank()
print(dataset_summary(bank))

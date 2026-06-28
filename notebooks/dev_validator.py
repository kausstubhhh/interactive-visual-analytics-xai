from src.data.loader import load_demo_dataset
from src.data.validator import dataset_summary, get_column_names, column_types

print("\nHELOC")
print("=" * 60)

heloc = load_demo_dataset("heloc")

print("\nSUMMARY")
print(dataset_summary(heloc))

print("\nFIRST 10 COLUMNS")
print(get_column_names(heloc)[:10])

print("\nCOLUMN TYPES")
print(column_types(heloc))


print("\nBANK")
print("=" * 60)

bank = load_demo_dataset("bank")

print("\nSUMMARY")
print(dataset_summary(bank))

print("\nFIRST 10 COLUMNS")
print(get_column_names(bank)[:10])

print("\nCOLUMN TYPES")
print(column_types(bank))

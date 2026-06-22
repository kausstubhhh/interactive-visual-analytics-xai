from src.data.loader import load_demo_dataset
from src.data.validator import dataset_summary
from src.data.validator import missing_values_report
from src.data.validator import column_types


print("\nHELOC")
print("=" * 50)

heloc = load_demo_dataset("heloc")

print(dataset_summary(heloc))

print("\nCOLUMN TYPES")
print(column_types(heloc))

print("\nMISSING VALUES")
print(missing_values_report(heloc))


print("\nBANK")
print("=" * 50)

bank = load_demo_dataset("bank")

print(dataset_summary(bank))
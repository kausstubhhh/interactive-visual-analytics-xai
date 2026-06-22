from src.data.loader import load_demo_dataset
from src.data.schema_detector import detect_schema


print("\nHELOC")
print("=" * 60)

heloc = load_demo_dataset("heloc")

heloc_schema = detect_schema(
    heloc,
    target_column="RiskPerformance"
)

print(heloc_schema)


print("\nBANK")
print("=" * 60)

bank = load_demo_dataset("bank")

bank_schema = detect_schema(
    bank,
    target_column="y"
)

print(bank_schema)
from src.data.loader import load_demo_dataset
from src.data.schema_detector import detect_schema
from src.data.preprocess import prepare_dataset


print("\nHELOC")
print("=" * 60)

heloc = load_demo_dataset("heloc")

heloc_schema = detect_schema(
    heloc,
    target_column="RiskPerformance"
)

(
    X_train,
    X_test,
    y_train,
    y_test,
    preprocessor
) = prepare_dataset(
    heloc,
    "RiskPerformance",
    heloc_schema
)

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)


print("\nBANK")
print("=" * 60)

bank = load_demo_dataset("bank")

bank_schema = detect_schema(
    bank,
    target_column="y"
)

(
    X_train,
    X_test,
    y_train,
    y_test,
    preprocessor
) = prepare_dataset(
    bank,
    "y",
    bank_schema
)

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)
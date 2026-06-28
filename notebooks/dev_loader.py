from src.data.loader import load_demo_dataset

print("TEST STARTED")

print("IMPORT SUCCESS")

print("\nLOADING HELOC")

heloc = load_demo_dataset("heloc")

print(heloc.shape)

print("\nLOADING BANK")

bank = load_demo_dataset("bank")

print(bank.shape)

print("\nSUCCESS")

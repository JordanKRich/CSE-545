import random

# Skip the first 7 lines
for _ in range(7):
    print()

# Generate and print out random coordinates
for city_number in range(1, 501):
    longitude = random.uniform(0, 180)
    latitude = random.uniform(0, 90)
    print(f"{city_number} {longitude:.6f} {latitude:.6f}")

import numpy as np
import pandas as pd

def create_time_array():
    """Create time array with 2-minute cadence for 30 days."""
    return np.arange(0, 30, 2/(24*60))

def generate_transit(time, period, depth, duration):
    """Generate transit signal."""
    phase = (time % period) / period
    transit_window = duration / (2 * period)
    return np.where((phase > 0.5 - transit_window) & (phase < 0.5 + transit_window), -depth, 0)

def generate_lightcurve(time, has_transit=True):
    """Generate single light curve."""
    period = np.random.uniform(3, 15)
    depth = np.random.uniform(0.01, 0.05) if has_transit else 0
    duration = np.random.uniform(0.1, 0.5)
    
    variability = 0.01 * np.sin(2 * np.pi * time / period)
    transit = generate_transit(time, period, depth, duration) if has_transit else 0
    noise = np.random.normal(0, 0.001, len(time))
    return 1 + variability + transit + noise

def create_dataset(time, num_curves):
    """Generate labeled dataset."""
    dataset = []
    labels = []
    for _ in range(num_curves):
        has_transit = np.random.rand() < 0.5
        dataset.append(generate_lightcurve(time, has_transit))
        labels.append(1 if has_transit else 0)
    return np.array(dataset), np.array(labels)

def validate_input(prompt, valid_options):
    """Force valid user input."""
    while True:
        try:
            choice = int(input(prompt))
            if choice in valid_options:
                return choice
            print(f"Invalid option. Choose from {valid_options}")
        except ValueError:
            print("Numbers only please")

if __name__ == "__main__":
    # Interactive size selection
    print("\nChoose dataset size:")
    print("1. 1000 samples\n2. 10000 samples\n3. 50000 samples")
    choice = validate_input("Enter option (1/2/3): ", [1, 2, 3])
    
    sizes = {1:1000, 2:10000, 3:50000}
    num_curves = sizes[choice]
    
    # Generate data
    time = create_time_array()
    X, y = create_dataset(time, num_curves)
    
    # Save to CSV
    filename = f"synthetic_exoplanet_dataset_{num_curves}.csv"
    pd.DataFrame(X, columns=[f"t_{i}" for i in range(len(time))]).assign(label=y).to_csv(filename, index=False)
    print(f"\nDataset with {num_curves} samples saved to {filename}")
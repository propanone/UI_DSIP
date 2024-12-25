import csv

def transform_data_to_csv(input_file, output_file):
    """
    Transform dat file with comma-separated key:value pairs into CSV format.
    
    Args:
        input_file (str): Path to input .dat file
        output_file (str): Path to output .csv file
    """
    # Store all rows and headers
    rows = []
    headers = set()
    
    # First pass: Read data and collect all possible headers
    with open(input_file, 'r') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                # Split the line into key-value pairs
                pairs = line.strip().split(',')
                
                # Convert to dictionary
                row_dict = {}
                for pair in pairs:
                    key, value = pair.split(':')
                    row_dict[key] = value
                    headers.add(key)
                
                rows.append(row_dict)
    
    # Convert headers set to sorted list for consistent column ordering
    headers = sorted(list(headers))
    
    # Write to CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

# Example usage
input_file = "datFileBis.dat"
output_file = "output.csv"

transform_data_to_csv(input_file, output_file)
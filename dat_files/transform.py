import csv

def transform_data_to_csv(input_file, output_file):
    rows = []
    headers = set()
    
    with open(input_file, 'r') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                # Split the line into key-value pairs
                pairs = line.strip().split(',')
                row_dict = {}
                for pair in pairs:
                    key, value = pair.split(':')
                    row_dict[key] = value
                    headers.add(key)
                
                rows.append(row_dict)
    
    headers = sorted(list(headers))
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

input_file = "client.dat"
output_file = "output.csv"
# remove the headers !
transform_data_to_csv(input_file, output_file)
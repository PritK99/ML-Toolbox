"""
The /ILSVRC/LOC_val_solution_cls.csv used in organize.py is derived from its original file ILSVRC/LOC_val_solution.csv. The original file also contains information like bounding boxes, which we don't require.
"""
import csv

# Original classes for validation set
input_csv = "/ssd_scratch/alexnet/LOC_val_solution.csv"

# Modified validation set
output_csv = "/ssd_scratch/alexnet/LOC_val_solution_cls.csv"

with open(input_csv, 'r') as infile, open(output_csv, 'w', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    writer.writerow(['ImageId', 'ClassId'])
    next(reader)
    for row in reader:
        img_id = row[0].strip()
        prediction_str = row[1].strip()
        primary_class = prediction_str.split()[0]  
        writer.writerow([img_id, primary_class])

print(f"Completed extraction")
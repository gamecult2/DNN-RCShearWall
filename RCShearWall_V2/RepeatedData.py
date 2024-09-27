import csv


def process_csv(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        unique_batches = set()
        batch = []
        total_batches = 0
        removed_batches = 0

        for i, row in enumerate(reader, 1):
            batch.append(row)

            if i % 3 == 0:  # Every third row
                total_batches += 1
                first_two_rows_tuple = tuple(map(tuple, batch[:2]))  # Convert first two rows to tuple
                if first_two_rows_tuple not in unique_batches:
                    unique_batches.add(first_two_rows_tuple)
                    writer.writerows(batch)  # Write the whole batch
                else:
                    removed_batches += 1
                batch = []  # Reset batch

            # if i % 3 == 0:  # Every third row
            #     total_batches += 1
            #     first_row_tuple = tuple(batch[0])  # Convert first row to tuple
            #     if first_row_tuple not in unique_batches:
            #         unique_batches.add(first_row_tuple)
            #         writer.writerows(batch)  # Write the whole batch
            #     else:
            #         removed_batches += 1
            #     batch = []  # Reset batch

        # Handle any remaining rows if the total is not divisible by 3
        if batch:
            total_batches += 1
            first_two_rows_tuple = tuple(map(tuple, batch[:2]))
            if first_two_rows_tuple not in unique_batches:
                writer.writerows(batch)
            else:
                removed_batches += 1

        # # Handle any remaining rows if the total is not divisible by 3
        # if batch:
        #     total_batches += 1
        #     first_row_tuple = tuple(batch[0])
        #     if first_row_tuple not in unique_batches:
        #         writer.writerows(batch)
        #     else:
        #         removed_batches += 1

    print(f"Processed data written to {output_file}")
    print(f"Total batches: {total_batches}")
    print(f"Batches removed: {removed_batches}")
    print(f"Batches kept: {total_batches - removed_batches}")


# Usage
input_file = "RCWall_Data/Original_Data/Original_Data.csv"
output_file = 'RCWall_Data/Processed_Data.csv'
process_csv(input_file, output_file)

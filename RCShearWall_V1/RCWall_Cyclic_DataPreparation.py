import csv


def open_csv_file(filename):
    """Opens a CSV file and returns a list of rows, where each row is a list of values."""
    with open(filename, "r") as f:
        return list(csv.reader(f))


def split_rows(rows):
    """Splits a list of rows into six lists, where each list contains the rows for a single data point."""
    return [rows[i:i + 6] for i in range(0, len(rows), 6)]


def extract_values(data_points, row_index):
    """Extracts values from a specific row of each data point and converts them to floats."""
    return [[float(value) for value in data_point[row_index][1:]] for data_point in data_points]


def save_data_to_file(filename, data):
    """Saves a list of data to a CSV file."""
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(data)


BASE_FILE_PATH = "RCWall_Data"  # Define a constant for the base file path

cyclic = False
pushover = False
both = True

if cyclic:
    filename = f"{BASE_FILE_PATH}/RCWall_Dataset_cyclic.csv"
    folder = "Dataset_cyclic"
elif pushover:
    filename = f"{BASE_FILE_PATH}/RCWall_Dataset_pushover.csv"
    folder = "Dataset_pushover"
elif both:
    filename = f"{BASE_FILE_PATH}/RCWall_Dataset_Full(ShortWall).csv"
    folder = "RCWall_Dataset_Full(ShortWall)"

# Read CSV file into a DataFrame
df = open_csv_file(filename)
data_points = split_rows(df)

if cyclic:
    # ------------------------ Inputs (Structural Parameters + Cyclic Loading) ---------------------------------------------------------------------
    save_data_to_file(f"{BASE_FILE_PATH}/{folder}/InputParameters.csv", extract_values(data_points, 0))
    # save_data_to_file(f"{BASE_FILE_PATH}/{folder}/InputCyclicDisplacement.csv", extract_values(data_points, 1))

    # ----------------------- Outputs (Hysteresis Curve - ShearBase Vs Lateral Displacement) -------------------------------------------------------
    # save_data_to_file(f"{BASE_FILE_PATH}/{folder}/OutputCyclicDisplacement.csv", extract_values(data_points, 2))
    # save_data_to_file(f"{BASE_FILE_PATH}/{folder}/OutputCyclicShear.csv", extract_values(data_points, 3))

if pushover:
    # ------------------------ Inputs (Structural Parameters + Cyclic Loading) ---------------------------------------------------------------------
    save_data_to_file(f"{BASE_FILE_PATH}/{folder}/InputParameters.csv", extract_values(data_points, 0))
    save_data_to_file(f"{BASE_FILE_PATH}/{folder}/InputPushoverDisplacement.csv", extract_values(data_points, 1))

    # ----------------------- Outputs (Pushover Curve -  ShearBase Vs Lateral Displacement) --------------------------------------------------------
    save_data_to_file(f"{BASE_FILE_PATH}/{folder}/OutputPushoverDisplacement.csv", extract_values(data_points, 2))
    save_data_to_file(f"{BASE_FILE_PATH}/{folder}/OutputPushoverShear.csv", extract_values(data_points, 3))

if both:
    # ------------------------ Inputs (Structural Parameters + Cyclic Loading) ---------------------------------------------------------------------
    save_data_to_file(f"{BASE_FILE_PATH}/{folder}/InputParameters.csv", extract_values(data_points, 0))
    save_data_to_file(f"{BASE_FILE_PATH}/{folder}/InputCyclicDisplacement.csv", extract_values(data_points, 1))
    save_data_to_file(f"{BASE_FILE_PATH}/{folder}/InputPushoverDisplacement.csv", extract_values(data_points, 4))

    # ----------------------- Outputs (Hysteresis Curve - ShearBase Vs Lateral Displacement) -------------------------------------------------------
    # save_data_to_file(f"{BASE_FILE_PATH}/{folder}/OutputCyclicDisplacement.csv", extract_values(data_points, 2))
    save_data_to_file(f"{BASE_FILE_PATH}/{folder}/OutputCyclicShear.csv", extract_values(data_points, 3))

    # ----------------------- Outputs (Pushover Curve -  ShearBase Vs Lateral Displacement) --------------------------------------------------------
    # save_data_to_file(f"{BASE_FILE_PATH}/{folder}/OutputPushoverDisplacement.csv", extract_values(data_points, 4))
    save_data_to_file(f"{BASE_FILE_PATH}/{folder}/OutputPushoverShear.csv", extract_values(data_points, 5))

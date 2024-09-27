"""
###########################################################################
# ReadSMDFile(in_filename, out_filename, dt)                                                           #
###########################################################################
# read gm input format
#
# Original Tcl Script:
#    Written by  : MHS
#    Date        : July 2000
# Translated to Python: elastropy.com (2023)
#
# A procedure which parses a ground motion record from the PEER
# strong motion database by finding dt in the record header, then
# echoing data values to the output file.
#
# Formal arguments
#   inFilename -- file which contains PEER strong motion record
#   outFilename -- file to be written in format G3 can read
#   dt -- time step determined from file header
#
# Assumptions
#   The header in the PEER record is, e.g., formatted as follows:
#    PACIFIC ENGINEERING AND ANALYSIS STRONG-MOTION DATA
#    IMPERIAL VALLEY 10/15/79 2319, EL CENTRO ARRAY 6, 230                           
#    ACCELERATION TIME HISTORY IN UNITS OF G                                         
#    NPTS=  3930, DT= .00500 SEC
"""
def ReadSMDFile(in_filename, out_filename, dt):
    # Open the input file and catch the error if it can't be read
    try:
        with open(in_filename, 'r') as in_file:
            # Open output file for writing
            with open(out_filename, 'w') as out_file:

                # Flag indicating dt is found and that ground motion
                # values should be read -- ASSUMES dt is on the last line
                # of the header!!!
                flag = False

                # Look at each line in the file
                for line in in_file:
                    if len(line.strip()) == 0:
                        # Blank line --> do nothing
                        continue
                    elif flag:
                        # Echo ground motion values to the output file
                        out_file.write(line)
                    else:
                        # Search header lines for dt
                        for word in line.split():
                            # Read in the time step
                            if flag:
                                dt[0] = float(word)
                                break
                            # Find the desired token and set the flag
                            if word == "DT=":
                                flag = True
                return True

    except IOError:
        print(f"Cannot open {in_filename} for reading")
        return False

# Example usage:
# in_filename = "../GMfiles/H-E12140.AT2"
# out_filename = "../GMfiles/H-E12140-2.g3"
# dt = [0.005]  # dt is passed by reference as a list to mimic Tcl's upvar
# ReadSMDFile(in_filename, out_filename, dt)
# print(f"Time step (DT): {dt[0]}")

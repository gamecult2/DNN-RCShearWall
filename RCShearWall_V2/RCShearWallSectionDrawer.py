import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# Read the parameters from the CSV file
csv_file = 'RCWall_Data/RCWall_Dataset_Fulltest.csv'
data = pd.read_csv(csv_file)


# Function to plot a 3D section of the RC shear wall
def plot_shear_wall_3d(tw, tb, hw, lw, lbe):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Main wall vertices
    wall_vertices = [
        [0, -tw/2, 0], [lw, -tw/2, 0], [lw, tw/2, 0], [0, tw/2, 0],  # Bottom face
        [0, -tw/2, hw], [lw, -tw/2, hw], [lw, tw/2, hw], [0, tw/2, hw]  # Top face
    ]

    # Left boundary element vertices
    left_be_vertices = [
        [0, -tb / 2, 0], [lbe, -tb / 2, 0], [lbe, tb / 2, 0], [0, tb / 2, 0],  # Bottom face
        [0, -tb / 2, hw], [lbe, -tb / 2, hw], [lbe, tb / 2, hw], [0, tb / 2, hw]  # Top face
    ]

    # Right boundary element vertices
    right_be_vertices = [
        [lw - lbe, -tb / 2, 0], [lw, -tb / 2, 0], [lw, tb / 2, 0], [lw - lbe, tb / 2, 0],  # Bottom face
        [lw - lbe, -tb / 2, hw], [lw, -tb / 2, hw], [lw, tb / 2, hw], [lw - lbe, tb / 2, hw]  # Top face
    ]

    # Define the faces of the cuboids
    faces = [
        [0, 1, 2, 3], [4, 5, 6, 7],  # Bottom and top faces
        [0, 1, 5, 4], [2, 3, 7, 6],  # Side faces
        [0, 3, 7, 4], [1, 2, 6, 5]  # Front and back faces
    ]

    # Plot main wall
    for face in faces:
        ax.add_collection3d(Poly3DCollection([[
            wall_vertices[vertex] for vertex in face
        ]], color='lightgray', edgecolor='black', alpha=0.5))

    # Plot left boundary element
    for face in faces:
        ax.add_collection3d(Poly3DCollection([[
            left_be_vertices[vertex] for vertex in face
        ]], color='gray', edgecolor='black', alpha=0.7))

    # Plot right boundary element
    for face in faces:
        ax.add_collection3d(Poly3DCollection([[
            right_be_vertices[vertex] for vertex in face
        ]], color='gray', edgecolor='black', alpha=0.7))

    # Set labels and title
    ax.set_xlabel('Length (mm)')
    ax.set_ylabel('Thickness (mm)')
    ax.set_zlabel('Height (mm)')
    ax.set_title('RC Shear Wall 3D Section')

    # Set limits and aspect ratio
    ax.set_xlim([0, lw])
    ax.set_ylim([0, max(tw, tb)])
    ax.set_zlim([0, hw])
    ax.set_box_aspect([lw, max(tw, tb), hw])  # Aspect ratio is 1:1:1

    # Show the plot
    plt.show()


# Function to plot a section of the RC shear wall
def plot_shear_wall_section(tw, tb, hw, lw, lbe):
    fig, ax = plt.subplots()

    # Plot the main wall section
    main_wall = plt.Rectangle((0, 0-(tw/2)), lw, tw, edgecolor='black', facecolor='lightgray')
    ax.add_patch(main_wall)

    # Plot the boundary elements
    boundary_element_left = plt.Rectangle((0, 0-(tb/2)), lbe, tb, edgecolor='black', facecolor='gray')
    boundary_element_right = plt.Rectangle((lw - lbe, 0-(tb/2)), lbe, tb, edgecolor='black', facecolor='gray')
    ax.add_patch(boundary_element_left)
    ax.add_patch(boundary_element_right)

    # Set the labels and title
    ax.set_xlabel('Length (mm)')
    ax.set_ylabel('Thickness (mm)')
    ax.set_title('RC Shear Wall Cross-Section')

    # Set the aspect ratio and limits
    ax.set_aspect('equal')
    ax.set_xlim([-lw * 0.1, lw * 1.1])
    ax.set_ylim([-tb * 1.5, tb * 1.5])

    # Add a grid for better visualization
    ax.grid(True)

    # Show the plot
    plt.show()


# Loop through each row in the CSV file and plot the shear wall section
for index, row in data.iterrows():
    tw = row['tw']
    tb = row['tb']
    hw = row['hw']
    lw = row['lw']
    lbe = row['lbe']
    print(tw)

    print(f"Plotting shear wall section for row {index + 1}")
    plot_shear_wall_section(tw, tb, hw, lw, lbe)
    # plot_shear_wall_3d(tw, tb, hw, lw, lbe)

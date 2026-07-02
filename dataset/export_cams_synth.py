#!/usr/bin/env python3
import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations, product

def get_frustum(cam_to_world, scale=0.25):
    """Generates 3D lines for a camera frustum based on its extrinsics."""
    w, h, d = scale * 0.8, scale * 0.5, scale
    pts_local = np.array([
        [0, 0, 0, 1],
        [-w, -h, d, 1],
        [ w, -h, d, 1],
        [ w,  h, d, 1],
        [-w,  h, d, 1]
    ]).T
    pts_world = cam_to_world @ pts_local
    pts = pts_world[:3, :]
    lines = [
        (pts[:,0], pts[:,1]), (pts[:,0], pts[:,2]), (pts[:,0], pts[:,3]), (pts[:,0], pts[:,4]),
        (pts[:,1], pts[:,2]), (pts[:,2], pts[:,3]), (pts[:,3], pts[:,4]), (pts[:,4], pts[:,1])
    ]
    return lines, pts[:,0]

def plot_box(ax):
    """Plots a clean 1x1x1 wireframe open box in the center."""
    r = [-0.5, 0.5]
    for s, e in combinations(np.array(list(product(r, r, r))), 2):
        if np.sum(np.abs(s-e)) == r[1]-r[0]:
            if s[2] == 0.5 and e[2] == 0.5:
                ax.plot3D(*zip(s, e), color="#FF5555", linestyle="--", linewidth=1.5, alpha=0.8)
            else:
                ax.plot3D(*zip(s, e), color="#FFFFFF", linewidth=2.0, alpha=0.8)

def main():
    parser = argparse.ArgumentParser(description="Generate 10 static camera view angles.")
    # Make sure this points to your actual JSON location
    parser.add_argument("--json", type=Path, default=Path("../data/synthetic_open_cube_15cams/calibration_dome.json"))
    parser.add_argument("--output-prefix", type=str, default="thesis_camera_angles_frame_", help="Prefix for saved images.")
    args = parser.parse_args()

    if not args.json.exists():
        print(f"File not found: {args.json}")
        return

    with open(args.json, 'r') as f:
        data = json.load(f)

    # --- Setup the Viewing Angles (Elevation, Azimuth in degrees) ---
    # This generates a spiral path around the scene
    angles_list = []
    for i in range(10):
        elev = 35 + i * 5 # Gradually increase elevation (35, 40, ..., 80)
        azim = 45 + i * 36 # Orbit 360 degrees (45, 81, ..., 369)
        angles_list.append((elev, azim % 360))

    # Output directory
    output_dir = Path("thesis_output_angles")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating 10 frames in: {output_dir}")

    # --- Setup Pretty Dark Theme ---
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(10, 8), facecolor='#121212')
    ax = fig.add_subplot(111, projection='3d', facecolor='#121212')
    
    # Keep axes off for all frames
    ax.set_axis_off()

    # Pre-render the static geometry
    # Draw the target box
    plot_box(ax)

    handles, labels = [], []

    # Parse and draw cameras (Static elements)
    for cam in data['cameras']:
        view_matrix = np.array(cam['extrinsics']['view_matrix']).reshape(4, 4)
        cam_to_world = np.linalg.inv(view_matrix)
        is_depth = cam['camera_type'] == 'depth'
        # c = '#FF00FF' if is_depth else '#00E5FF' # High-contrast Magenta/Cyan
        c = '#FF00FF' if is_depth else '#00E5FF' 
        lbl = 'Depth Camera' if is_depth else 'Color Camera'
        lines, center = get_frustum(cam_to_world, scale=0.3) # Slightly smaller scale
        for start, end in lines:
            ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color=c, lw=1.5, alpha=0.6)
        scatter = ax.scatter(*center, c=c, s=30, edgecolors='white', linewidths=0.5, label=lbl if lbl not in labels else "")
        # Label the camera ID
        ax.text(center[0], center[1], center[2] + 0.15, cam['camera_id'], size=8, color=c, fontweight='bold', ha='center')
        if lbl not in labels:
            handles.append(scatter)
            labels.append(lbl)

    # Calculate and set equal aspect ratio bounds (Static limits)
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    # Increase viewing range to see cameras pushed 3m back
    radius = max(radius, 3.2)
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])
    
    # Legend
    ax.legend(handles, labels, facecolor='#222222', edgecolor='none', loc='upper right', fontsize=11)

    # --- Loop through angles and SAVE each frame ---
    for i, (elev, azim) in enumerate(angles_list):
        # Update the viewing angle
        ax.view_init(elev=elev, azim=azim)
        
        # Format the output path
        output_filename = f"{args.output_prefix}{i:02d}.png"
        output_path = output_dir / output_filename
        
        # Save the high-res output
        plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
        print(f"Saved: {output_filename} (Elev: {elev}, Azim: {azim})")

    plt.close(fig) # Close the figure to free memory
    print("Done generating all angles.")

if __name__ == "__main__":
    main()
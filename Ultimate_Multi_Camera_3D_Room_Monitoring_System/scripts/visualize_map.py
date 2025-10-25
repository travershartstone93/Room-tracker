import open3d as o3d
import sys

if len(sys.argv) < 2:
    print("Usage: python visualize_map.py room_map.ply")
    sys.exit(1)

pcd = o3d.io.read_point_cloud(sys.argv[1])
o3d.visualization.draw_geometries([pcd])

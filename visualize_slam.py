import numpy as np
import open3d as o3d
import numpy as np

class SLAMVisualizer:
    def __init__(self):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        
        # Set up rendering options
        opt = self.vis.get_render_option()
        opt.background_color = np.asarray([0.1, 0.1, 0.1])  # Dark background
        opt.point_size = 1.0
        
        # Set up camera view
        self.view_control = self.vis.get_view_control()
    
    def load_reconstruction(self, pointcloud_path, trajectory_path=None):
        """Load the point cloud and camera trajectory."""
        # Load and process point cloud
        self.pcd = o3d.io.read_point_cloud(pointcloud_path)
        
        # Estimate normals for better visualization
        self.pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.1, max_nn=30))
        
        # Add point cloud to visualizer
        self.vis.add_geometry(self.pcd)
        
        # Load and visualize camera trajectory if available
        if trajectory_path:
            trajectory = np.load(trajectory_path)
            self.visualize_camera_path(trajectory)
    
    def visualize_camera_path(self, trajectory):
        """Create a line set showing the camera path."""
        # Create line set from trajectory points
        lines = [[i, i+1] for i in range(len(trajectory)-1)]
        colors = [[1, 0, 0] for _ in range(len(lines))]  # Red lines
        
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(trajectory)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        
        # Add camera frustums at key points
        for i in range(0, len(trajectory), 20):  # Add frustum every 20th position
            frustum = self.create_camera_frustum(trajectory[i])
            self.vis.add_geometry(frustum)
        
        self.vis.add_geometry(line_set)
    
    def create_camera_frustum(self, position, size=0.5):
        """Create a simple camera frustum visualization."""
        points = np.array([
            [0, 0, 0],  # Camera center
            [-size, -size, size],  # Front corners
            [size, -size, size],
            [size, size, size],
            [-size, size, size]
        ]) + position
        
        lines = [[0, 1], [0, 2], [0, 3], [0, 4],  # Lines from center to corners
                [1, 2], [2, 3], [3, 4], [4, 1]]    # Lines connecting corners
        
        colors = [[0, 1, 0] for _ in range(len(lines))]  # Green frustum
        
        frustum = o3d.geometry.LineSet()
        frustum.points = o3d.utility.Vector3dVector(points)
        frustum.lines = o3d.utility.Vector2iVector(lines)
        frustum.colors = o3d.utility.Vector3dVector(colors)
        
        return frustum
    
    def add_coordinate_frame(self, size=1.0):
        """Add a coordinate frame to help with orientation."""
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
        self.vis.add_geometry(coord_frame)
    
    def run(self):
        """Start the visualization."""
        print("Visualization Controls:")
        print("Left click + drag: Rotate")
        print("Right click + drag: Pan")
        print("Mouse wheel: Zoom")
        print("Press 'h' to return to default view")
        print("Press 'r' to reset camera view")
        print("Press 'q' to exit")
        
        self.vis.run()
        self.vis.destroy_window()

def visualize_reconstruction(pointcloud_path, trajectory_path=None):
    """
    Visualize the 3D reconstruction and camera trajectory
    
    Args:
        pointcloud_path (str): Path to the point cloud file (.ply)
        trajectory_path (str): Path to the trajectory file (.npy)
    """
    visualizer = SLAMVisualizer()
    visualizer.load_reconstruction(pointcloud_path, trajectory_path)
    visualizer.add_coordinate_frame()
    visualizer.run()

# Example usage
if __name__ == "__main__":
    # Replace these with your actual file paths
    pointcloud_file = "output_reconstruction.ply"
    trajectory_file = "output_reconstruction_trajectory.npy"
    
    visualize_reconstruction(pointcloud_file, trajectory_file)
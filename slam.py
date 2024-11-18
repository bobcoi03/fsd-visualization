import cv2
import numpy as np
from pathlib import Path
import open3d as o3d
from tqdm import tqdm

class DashcamSLAM:
    def __init__(self):
        # Initialize ORB feature detector
        self.orb = cv2.ORB_create(nfeatures=3000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Camera parameters (adjust these based on your dashcam)
        self.focal_length = 718.8560  # typical focal length
        self.pp = (607.1928, 185.2157)  # principal point
        self.K = np.array([
            [self.focal_length, 0, self.pp[0]],
            [0, self.focal_length, self.pp[1]],
            [0, 0, 1]
        ])
        
        # Storage for 3D points and camera positions
        self.points3D = []
        self.camera_positions = []
        
    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        prev_frame = None
        prev_kp = None
        prev_des = None
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect ORB features
            kp, des = self.orb.detectAndCompute(gray, None)
            
            if prev_frame is not None:
                # Match features
                matches = self.bf.match(prev_des, des)
                matches = sorted(matches, key=lambda x: x.distance)
                
                # Get matched point coordinates
                pts1 = np.float32([prev_kp[m.queryIdx].pt for m in matches])
                pts2 = np.float32([kp[m.trainIdx].pt for m in matches])
                
                # Calculate essential matrix
                E, mask = cv2.findEssentialMat(pts1, pts2, self.K)
                
                # Recover pose
                _, R, t, mask = cv2.recoverPose(E, pts1, pts2, self.K)
                
                # Triangulate points
                P1 = self.K @ np.hstack((np.eye(3), np.zeros((3,1))))
                P2 = self.K @ np.hstack((R, t))
                
                pts4D = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
                pts3D = pts4D[:3] / pts4D[3]
                
                # Store results
                self.points3D.extend(pts3D.T)
                self.camera_positions.append(t.flatten())
            
            prev_frame = gray
            prev_kp = kp
            prev_des = des
            
        cap.release()
        
    def save_pointcloud(self, output_path):
        # Convert to Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(self.points3D))
        
        # Optional: Remove outliers
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        
        # Save the point cloud
        o3d.io.write_point_cloud(output_path, pcd)
        
        # Save camera trajectory
        np.save(output_path.replace('.ply', '_trajectory.npy'), 
                np.array(self.camera_positions))

def process_dashcam_footage(video_path, output_path):
    """
    Process dashcam footage to create a 3D reconstruction
    
    Args:
        video_path (str): Path to input video file
        output_path (str): Path to save the output point cloud
    """
    slam = DashcamSLAM()
    
    print("Processing video...")
    slam.process_video(video_path)
    
    print("Saving 3D reconstruction...")
    slam.save_pointcloud(output_path)
    
    print(f"Reconstruction complete! Results saved to {output_path}")

if __name__ == "__main__":
    video_path = "drive_trimmed_480p.mp4"
    output_path = "output_reconstruction.ply"
    process_dashcam_footage(video_path, output_path)
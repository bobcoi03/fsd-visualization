import cv2
import sys
import numpy as np

def play_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video file")
        return
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"Video Properties:")
    print(f"Resolution: {frame_width}x{frame_height}")
    print(f"FPS: {fps}")
    
    cv2.namedWindow('Video Player', cv2.WINDOW_NORMAL)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video file")
            break

        copy = np.copy(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        height, width = frame.shape[:2]
        mask = np.zeros_like(edges)
        polygon = np.array([
            [
                (0, height),
                (width // 2, height // 2),
                (width, height),
            ]
        ], np.int32)
        cv2.fillPoly(mask, polygon, 255)
        masked_edges = cv2.bitwise_and(edges, mask)

        lines = cv2.HoughLinesP(
            masked_edges,
            rho=2,
            theta=np.pi/180,
            threshold=40,
            minLineLength=40,
            maxLineGap=5,
        )

        if lines is not None:
            averaged_lines = average(copy, lines)
            if averaged_lines is not None:
                black_lines = display_lines(copy, averaged_lines)
                if black_lines is not None:  # Add check for None return
                    lanes = cv2.addWeighted(copy, 0.8, black_lines, 1, 1)
                    cv2.imshow("Video Player", lanes)
                else:
                    cv2.imshow("Video Player", copy)
            else:
                cv2.imshow("Video Player", copy)
        else:
            cv2.imshow("Video Player", copy)

        if cv2.waitKey(1000//fps) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def average(image, lines):
    left = []
    right = []
    
    if lines is None:
        return None
        
    for line in lines:
        if line is None or len(line) == 0:
            continue
            
        x1, y1, x2, y2 = line.reshape(4)
        
        # Avoid division by zero
        if x2 - x1 == 0:
            continue
            
        # Fit polynomial of degree 1 (linear equation) to the points
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        y_int = parameters[1]
        
        if slope < 0:
            left.append((slope, y_int))
        else:
            right.append((slope, y_int))
    
    # Add safety checks for empty lists
    if len(left) == 0 or len(right) == 0:
        return None
    
    # Average the lines on each side
    left_avg = np.average(left, axis=0)
    right_avg = np.average(right, axis=0)
    
    try:
        # Convert averaged lines to points
        left_line = make_points(image, left_avg)
        right_line = make_points(image, right_avg)
        
        if left_line is None or right_line is None:
            return None
            
        return np.array([left_line, right_line])
    except:
        return None

def make_points(image, average):
    try:
        slope, y_int = average
        y1 = image.shape[0]
        y2 = int(y1 * (3/5))
        
        # Avoid division by zero
        if slope == 0:
            return None
            
        # Fix the x-coordinate calculation
        x1 = int((y1 - y_int) / slope)
        x2 = int((y2 - y_int) / slope)
        
        # Validate coordinates are within image bounds
        height, width = image.shape[:2]
        if not (0 <= x1 <= width and 0 <= y1 <= height and 
                0 <= x2 <= width and 0 <= y2 <= height):
            return None
            
        return np.array([x1, y1, x2, y2], dtype=np.int32)
    except:
        return None

def display_lines(image, lines):
    try:
        lines_image = np.zeros_like(image)
        if lines is not None:
            for line in lines:
                if line is None or len(line) != 4:
                    continue
                    
                x1, y1, x2, y2 = line
                
                # Ensure coordinates are integers
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                
                # Validate coordinates
                height, width = image.shape[:2]
                if not (0 <= x1 <= width and 0 <= y1 <= height and 
                        0 <= x2 <= width and 0 <= y2 <= height):
                    continue
                    
                cv2.line(lines_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
        return lines_image
    except:
        return None

if __name__ == "__main__":
    video_path = "drive_480p_new.mp4"
    play_video(video_path)
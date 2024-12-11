import cv2
import numpy as np
import sys

def get_edge_colors(image):
    """Get median colors for each edge of the image"""
    h, w = image.shape[:2]

    # Get 10-pixel wide strips from each edge
    top_colors = image[0:10, :].reshape(-1, 3)
    bottom_colors = image[h-10:h, :].reshape(-1, 3)
    left_colors = image[:, 0:10].reshape(-1, 3)
    right_colors = image[:, w-10:w, :].reshape(-1, 3)

    # Calculate median color for each edge
    top_median = np.median(top_colors, axis=0).astype(np.uint8)
    bottom_median = np.median(bottom_colors, axis=0).astype(np.uint8)
    left_median = np.median(left_colors, axis=0).astype(np.uint8)
    right_median = np.median(right_colors, axis=0).astype(np.uint8)

    return top_median, bottom_median, left_median, right_median

def create_flood_fill_mask(image):
    """Create mask using flood fill from edge points"""
    h, w = image.shape[:2]

    # Get median colors for each edge
    top_color, bottom_color, left_color, right_color = get_edge_colors(image)

    # Create mask with extra border pixels for flood fill
    mask = np.zeros((h+2, w+2), np.uint8)

    # Set up flood fill flags
    floodflags = 4 # 4-connected neighbors
    floodflags |= cv2.FLOODFILL_MASK_ONLY
    floodflags |= (255 << 9)

    # Generate seed points along each edge
    points_per_edge = 100
    x_points = np.linspace(10, w-10, points_per_edge).astype(np.int32)
    y_points = np.linspace(10, h-10, points_per_edge).astype(np.int32)

    # Flood fill from each edge
    lo = (12,)*3
    hi = (12,)*3
    for x in x_points:
        # Top edge
        cv2.floodFill(image, mask, (x, 3), (255,0,0),
                      lo, hi, floodflags)
        cv2.floodFill(image, mask, (x, 8), (255,0,0),
                      lo, hi, floodflags)
        # Bottom edge
        cv2.floodFill(image, mask, (x, h-3), (255,0,0),
                      lo, hi, floodflags)
        cv2.floodFill(image, mask, (x, h-8), (255,0,0),
                      lo, hi, floodflags)

    for y in y_points:
        # Left edge
        cv2.floodFill(image, mask, (3, y), (255,0,0),
                      lo, hi, floodflags)
        cv2.floodFill(image, mask, (8, y), (255,0,0),
                      lo, hi, floodflags)
        # Right edge
        cv2.floodFill(image, mask, (w-3, y), (255,0,0),
                      lo, hi, floodflags)
        cv2.floodFill(image, mask, (w-8, y), (255,0,0),
                      lo, hi, floodflags)

    # Remove the extra border from the mask
    mask = mask[1:-1, 1:-1]

    # Clean up the mask
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask

def intersections(edged):
    """Find intersections of Hough lines"""
    h, w = edged.shape

    # Get Hough lines
    lines = cv2.HoughLines(edged, 2, np.pi/180, 250)
    if lines is None:
        return None, None

    # Number of lines
    n = lines.shape[0]
    print(f"Number of lines: {n}")

    # Matrix with the values of cos(theta) and sin(theta) for each line
    T = np.zeros((n, 2), dtype=np.float32)
    # Vector with values of rho
    R = np.zeros(n, dtype=np.float32)

    # Fill matrices with cos(theta), sin(theta), and rho values
    T[:, 0] = np.cos(lines[:, 0, 1])  # cos(theta)
    T[:, 1] = np.sin(lines[:, 0, 1])  # sin(theta)
    R = lines[:, 0, 0]  # rho values

    # Number of combinations of all lines
    c = int(n * (n-1) / 2)
    # Matrix with the obtained intersections (x, y)
    XY = np.zeros((c, 2))

    # Finding intersections between all lines
    intersection_idx = 0
    for i in range(n):
        for j in range(i+1, n):
            try:
                XY[intersection_idx] = np.linalg.inv(T[[i,j]]).dot(R[[i,j]])
                intersection_idx += 1
            except np.linalg.LinAlgError:
                continue

    # Trim XY matrix to only include valid intersections
    XY = XY[:intersection_idx]

    # Filtering out coordinates outside the photo (with 20-pixel margin)
    margin = 20
    mask = ((XY[:, 0] > -margin) &
            (XY[:, 0] <= w + margin) &
            (XY[:, 1] > -margin) &
            (XY[:, 1] <= h + margin))
    XY = XY[mask]

    return XY, lines

def order_points(pts):
    """Order points in clockwise order starting from top-left"""
    if len(pts) < 4:
        return None

    rect = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def draw_debug_visualization(image, lines, mask, corners, edge_colors=None):
    """Draw debug visualization showing mask, corners, and edge colors"""
    debug_img = image.copy()

    # Draw all detected lines
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            cv2.line(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw mask as semi-transparent overlay
    overlay = np.zeros_like(debug_img)
    overlay[mask > 0] = [0, 255, 0]  # Green for mask
    debug_img = cv2.addWeighted(debug_img, 1, overlay, 0.3, 0)

    # Draw corner points and connecting lines
    if corners is not None:
        for corner in corners:
            cv2.circle(debug_img, tuple(corner.astype(int)), 8, (255, 0, 255), -1)

        #for i in range(4):
        #    pt1 = tuple(corners[i].astype(int))
        #    pt2 = tuple(corners[(i + 1) % 4].astype(int))
        #    cv2.line(debug_img, pt1, pt2, (0, 255, 0), 2)

    # Draw edge colors if provided
    if edge_colors is not None:
        top, bottom, left, right = edge_colors
        h, w = image.shape[:2]
        cv2.rectangle(debug_img, (w//2-50, 0), (w//2+50, 20), tuple(map(int, top)), -1)
        cv2.rectangle(debug_img, (w//2-50, h-20), (w//2+50, h), tuple(map(int, bottom)), -1)
        cv2.rectangle(debug_img, (0, h//2-50), (20, h//2+50), tuple(map(int, left)), -1)
        cv2.rectangle(debug_img, (w-20, h//2-50), (w, h//2+50), tuple(map(int, right)), -1)

    return debug_img

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <image_path>")
        sys.exit(1)

    # Read image
    image_path = sys.argv[1]
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        sys.exit(1)

    # Get edge colors and create mask using flood fill
    edge_colors = get_edge_colors(image)
    mask = create_flood_fill_mask(image)
    cv2.imshow('Mask', mask)

    # Find edges in the mask
    edges = cv2.Canny(mask, 50, 150)
    cv2.imshow('Edges', edges)

    # Find intersections
    points, lines = intersections(edges)
    if points is None or len(points) < 4:
        print("Error: Could not find enough intersection points")
        cv2.imshow('Debug', draw_debug_visualization(image, lines, mask, None, edge_colors))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        sys.exit(1)

    # Order points
    corners = order_points(points)
    if corners is None:
        print("Error: Could not order corner points")
        sys.exit(1)

    # Calculate output dimensions
    width_1 = np.sqrt(((corners[1][0] - corners[0][0]) ** 2) +
                     ((corners[1][1] - corners[0][1]) ** 2))
    width_2 = np.sqrt(((corners[2][0] - corners[3][0]) ** 2) +
                     ((corners[2][1] - corners[3][1]) ** 2))
    max_width = max(int(width_1), int(width_2))

    height_1 = np.sqrt(((corners[3][0] - corners[0][0]) ** 2) +
                      ((corners[3][1] - corners[0][1]) ** 2))
    height_2 = np.sqrt(((corners[2][0] - corners[1][0]) ** 2) +
                      ((corners[2][1] - corners[1][1]) ** 2))
    max_height = max(int(height_1), int(height_2))

    # Define destination points
    dst_points = np.float32([[0, 0],
                           [max_width - 1, 0],
                           [max_width - 1, max_height - 1],
                           [0, max_height - 1]])

    # Apply perspective transform
    matrix = cv2.getPerspectiveTransform(corners, dst_points)
    result = cv2.warpPerspective(image, matrix, (max_width, max_height))

    # Resize result for correct aspect-ratio
    result = cv2.resize(result, ((max_height * 16) // 9, max_height))

    # Show results
    cv2.imshow('Original', image)
    cv2.imshow('Debug', draw_debug_visualization(image, lines, mask, corners, edge_colors))
    cv2.imshow('Corrected', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
#Imports
#region imoprts
import time
import numpy as np
import cv2
from tqdm import trange
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(); sns.set_style('dark')
from ipywidgets import interact, interactive, fixed, interact_manual

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
#endregion


# Functions
#region functions
def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def disp(img, title='', s=8, vmin=None, vmax=None, write=False, file_name=None):
    plt.figure(figsize=(s,s))
    plt.axis('off')
    if vmin is not None and vmax is not None:
        plt.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    if write and file_name is not None:
        plt.savefig(file_name)
    plt.show()

class GHD_Scaler:
    def __init__(self):
        pass

    def fit(self, X):
        pass

    def transform(self, X):
        return X

    def inverse_transform(self, X):
        return X

def cluster_points(points, rotate_degree=0, squash_factor=100, n_clusters=7, debug=False):
    X = points.copy()

    if rotate_degree!=0:
        # Let's Rotate the points by 'rotate_degree' degrees
        theta = rotate_degree*np.pi/180

        # Define the rotation matrix
        R = np.array([
            [np.cos(theta), np.sin(theta)],
            [-np.sin(theta), np.cos(theta)]])

        # Rotate the points
        X = X @ R.T

    X[:,1] /= squash_factor

    cluster_ids = KMeans(n_clusters=n_clusters, random_state=1, algorithm='auto', max_iter=20, n_init=1).fit_predict(X)

    if debug:
        plt.figure(figsize=(6, 6))
        plt.scatter(X[:, 0], X[:, 1], c=cluster_ids, cmap='plasma_r')
        plt.title(f"Positive Lines (Rotated by {rotate_degree} degrees and scaled y by {squash_factor})")
        plt.show()

    return cluster_ids

def line_to_points(lines):
    points = np.zeros((len(lines)*2, 2))
    for i in range(len(lines)):
        # point A on a line
        points[2*i][0] = (lines[i][0])
        points[2*i][1] = (lines[i][1])
        # point B on a line
        points[2*i+1][0] = (lines[i][2])
        points[2*i+1][1] = (lines[i][3])
    return points

def display_reconstructed_faces(reconstructed_faces):
    faces_names = ["Left", "Right", "Top"]
    for f in range(3):
        try:
            cv2.imshow(f"{faces_names[f]} View", bgr(cv2.resize(reconstructed_faces[f], (200, 200), interpolation=cv2.INTER_NEAREST)))
        except:
            pass

def plot_lines_on_cube(points_1, points_2, points_3, fitted_ms_all, fitted_bs_all, scalers_all,  img_gray_rgb, y_pred_1, y_pred_2, y_pred_3, direction_list=[0,1,2], n_clusters=7):
    plt.figure(figsize=(6, 6))
    plt.imshow(img_gray_rgb)

    for i in direction_list:
        for cluster_id in range(n_clusters):
            # Get the points of this cluster
            if i==0:
                X = points_1.copy()[y_pred_1==cluster_id]
            elif i==1:
                X = points_2.copy()[y_pred_2==cluster_id]
            elif i==2:
                X = points_3.copy()[y_pred_3==cluster_id]
            
            # Get the corresponding scaler
            scaler = scalers_all[i][cluster_id]

            # Calculate the points in the current line
            x = np.arange(0,img_gray_rgb.shape[1])

            # Scale the x values so that they work with m and b
            x = scaler.transform(np.repeat(x[:,None], 2, axis=1))[:,0]
            y = fitted_ms_all[i][cluster_id]*x+fitted_bs_all[i][cluster_id]

            # Concatenate fitted line's x and y
            if i==0:
                # if vertical: x=my+b
                line_X = np.column_stack([y,x])
            else:
                # else: y=mx+b
                line_X = np.column_stack([x,y])
            
            # Inverse Scaler transform
            line_X = scaler.inverse_transform(line_X)

            plt.plot(line_X[:, 0], line_X[:, 1], c=colors_01[i], linewidth=1.5)

    plt.ylim([0,img_gray_rgb.shape[0]])
    plt.xlim([0,img_gray_rgb.shape[1]])
    plt.gca().invert_yaxis()
    plt.title("Fitted lines")

    plt.show()

def plot_intersection_points_on_cube(fitted_ms_all, fitted_bs_all, scalers_all,  img_gray_rgb, direction_list=[1,2], debug=True, c="g"):
    points_on_the_face = []

    if debug:
        plt.figure(figsize=(6, 6))
        plt.imshow(img_gray_rgb)

    # Sort lines from left to right and bottom to top
    msi = fitted_ms_all[direction_list[0]]
    bsi = fitted_bs_all[direction_list[0]]
    scaler_i = scalers_all[direction_list[0]]

    # Sort lines by b
    if direction_list[0] == 0 and direction_list[1] == 2:
        sorted_indices_i = np.argsort(bsi)[::-1]
    elif direction_list[0] == 0 and direction_list[1] == 1:
        sorted_indices_i = np.argsort(bsi)
    else:
        sorted_indices_i = np.argsort(bsi)

    msi = np.array(msi)[sorted_indices_i]
    bsi = np.array(bsi)[sorted_indices_i]
    scaler_i = np.array(scaler_i)[sorted_indices_i]
    
    msj = fitted_ms_all[direction_list[1]]
    bsj = fitted_bs_all[direction_list[1]]
    scaler_j= scalers_all[direction_list[1]]

    # sort lines by b
    if direction_list[0] == 0 and direction_list[1] == 2:
        sorted_indices_j = np.argsort(bsj)[::-1]
    elif direction_list[0] == 0 and direction_list[1] == 1:
        sorted_indices_j = np.argsort(bsj)[::-1]
    else:
        sorted_indices_j = np.argsort(bsj)

    msj = np.array(msj)[sorted_indices_j]
    bsj = np.array(bsj)[sorted_indices_j]
    scaler_j = np.array(scaler_j)[sorted_indices_j]


    # first 4 lines of dir_a
    for i in range(4):
        # first 4 lines of dir_b
        for j in range(4):
            m1 = msi[i]
            b1 = bsi[i]
            m2 = msj[j]
            b2 = bsj[j]
            if direction_list[0] == 0:
                b1 = -b1/m1
                m1 = 1/m1
            
            x = (b2-b1)/(m1-m2)
            y = m1*x+b1

            points_on_the_face.append([x,y])

            if debug:
                plt.scatter([x], [y], c=c)

    if debug:
        plt.ylim([0,img_gray_rgb.shape[0]])
        plt.xlim([0,img_gray_rgb.shape[1]])
        plt.gca().invert_yaxis()
        plt.title("Fitted lines")

        plt.show()

    return points_on_the_face


def fit_lines(points, y_pred, n_clusters=7, is_vertical=False):
    fitted_ms = []
    fitted_bs = []
    scalers = []

    for cluster_id in range(n_clusters):
        X = points.copy()[y_pred==cluster_id]

        # Scale features
        scaler = GHD_Scaler()
        scaler.fit(X)
        X = scaler.transform(X)
        scalers.append(scaler)

        # Fit a line to the points using linear regression
        regr = LinearRegression()

        # If 'is_vertical': fit x=my+b instead of y=mx+b
        if is_vertical:
            regr.fit(X[:,1].reshape(-1, 1), X[:,0])
        else:
            regr.fit(X[:,0].reshape(-1, 1), X[:,1])

        m = regr.coef_[0]
        b = regr.intercept_

        fitted_ms.append(m)
        fitted_bs.append(b)
    
    return fitted_ms, fitted_bs, scalers


def extract_faces(img, kernel_size=5, canny_low=0, canny_high=75, min_line_length=40, max_line_gap=20, center_sampling_width=10, colors=None, colors_01=None, n_clusters=7, debug=False, debug_time=True):
    """Takes an image of a rubiks cube, finds edges, fits lines to edges and extracts the faces

    Args:
        img (RGB image): rubiks cube image
    """
    start_time = time.perf_counter_ns()

    # 1. Convert to Gray
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    

    # 2. Blur
    blur_gray = cv2.GaussianBlur(img_gray, (kernel_size, kernel_size), 0)


    # 3. Canny
    edges = cv2.Canny(blur_gray, canny_low, canny_high)

    # 4. HoughLinesP
    # Distance resolution in pixels of the Hough grid
    rho = 1
    # Angular resolution in radians of the Hough grid
    theta = np.pi / 180
    # Other Hough params
    threshold = 15
    min_line_length = min_line_length
    max_line_gap = max_line_gap
    line_image = np.copy(img) * 0

    # Run Hough on edge detected image
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

    # 5. Calculate Angles
    if lines is not None:
        # print(len(lines),"lines detected")

        angles = []
        for line in lines:
            for x1, y1, x2, y2 in line:
                # Calculate the angle
                angle = cv2.fastAtan2(float(y2-y1), float(x2-x1))

                # 210 == 180+30 ==> 30
                if angle >= 180:
                    angle -= 180
                angles.append(angle)

        angles = np.array(angles)

        # Cluster the angles to find the breaking points
        angles_clustering = KMeans(n_clusters=3, n_init=2)
        angles_clustering.fit(angles.reshape(-1, 1))
        sorted_centers = sorted(angles_clustering.cluster_centers_)

        # Used to split lines into vertical/+ve/-ve categories
        angle_limits = [0, 60, 120, 180] 
        angle_limits[1] = ((sorted_centers[0] + sorted_centers[1]) // 2)[0]
        angle_limits[2] = ((sorted_centers[1] + sorted_centers[2]) // 2)[0]

        # used in step 7 to orient the clusters
        rotation_angles = [0, 60, 120] 
        rotation_angles[0] = -(90 - sorted_centers[1])[0] # vertical
        rotation_angles[1] = -(90 - sorted_centers[0])[0] # positive
        rotation_angles[2] = -(90 - sorted_centers[2])[0] # negative


        # if debug:
        #     print("Cluster centers:")
        #     print([int(center[0]) for center in sorted_centers])
        #     print("Breaking angles:")
        #     print(angle_limits)
        #     print("Rotation angles:")
        #     print(rotation_angles)

        # if debug_time:
        #     print("5. Calc Angles:\t\t",
        #         (time.perf_counter_ns() - start_time)/1000000, "ms")
        #     start_time = time.perf_counter_ns()

        # 5.5 Calculate Angles
        angles = []
        vertical_lines = []
        negative_lines = []
        positive_lines = []
        for line in lines:
            for x1, y1, x2, y2 in line:
                # Calculate the angle
                angle = cv2.fastAtan2(float(y2-y1), float(x2-x1))
                angles.append(angle)

                # 210 == 180+30 == 30
                if angle >= 180:
                    angle -= 180
                angles.append(angle)

                # Find the type of the line
                if angle_limits[0] <= angle <= angle_limits[1]:
                    cluster_id = 0
                    positive_lines.append([x1, y1, x2, y2])
                if angle_limits[1] <= angle <= angle_limits[2]:
                    cluster_id = 1
                    vertical_lines.append([x1, y1, x2, y2])
                if angle_limits[2] <= angle <= angle_limits[3]:
                    cluster_id = 2
                    negative_lines.append([x1, y1, x2, y2])

                cv2.line(line_image, (x1, y1), (x2, y2), colors[cluster_id], 5)

        cv2.imshow("DEBUG", line_image)
        # return None

        # 6. Lines to points
        points_1 = line_to_points(vertical_lines)
        points_2 = line_to_points(positive_lines)
        points_3 = line_to_points(negative_lines)

        # 7. Cluster points
        y_pred_1 = cluster_points(
            points_1, rotate_degree=rotation_angles[0], squash_factor=100, debug=debug)
        y_pred_2 = cluster_points(
            points_2, rotate_degree=rotation_angles[1], squash_factor=100, debug=debug)
        y_pred_3 = cluster_points(
            points_3, rotate_degree=rotation_angles[2], squash_factor=100, debug=debug)

        # 8. Line Fitting (y=mx+b) or (x=my+b)
        fitted_ms_1, fitted_bs_1, scalers_1 = fit_lines(
            points_1, y_pred_1, n_clusters=n_clusters, is_vertical=True)
        fitted_ms_2, fitted_bs_2, scalers_2 = fit_lines(
            points_2, y_pred_2, n_clusters=n_clusters, is_vertical=False)
        fitted_ms_3, fitted_bs_3, scalers_3 = fit_lines(
            points_3, y_pred_3, n_clusters=n_clusters, is_vertical=False)

        fitted_ms_all = [fitted_ms_1, fitted_ms_2, fitted_ms_3]
        fitted_bs_all = [fitted_bs_1, fitted_bs_2, fitted_bs_3]
        scalers_all = [scalers_1, scalers_2, scalers_3]

        # 10. Find intersection points

        img_gray_rgb = np.repeat(img_gray[:, :, None], 3, 2)
        points_on_left_face = plot_intersection_points_on_cube(
            fitted_ms_all, fitted_bs_all, scalers_all,  img_gray_rgb,
            direction_list=[0, 1], debug=debug)

        points_on_right_face = plot_intersection_points_on_cube(
            fitted_ms_all, fitted_bs_all, scalers_all,  img_gray_rgb,
            direction_list=[0, 2], debug=debug)

        points_on_top_face = plot_intersection_points_on_cube(
            fitted_ms_all, fitted_bs_all, scalers_all,  img_gray_rgb,
            direction_list=[1, 2], debug=debug)

        # 11. Find face centers
        points_left = np.array(points_on_left_face)
        points_right = np.array(points_on_right_face)
        points_top = np.array(points_on_top_face)

        face_indices = [
            [2, 3, 7, 6], [6, 7, 11, 10], [10, 11, 15, 14],
            [1, 2, 6, 5], [5, 6, 10, 9], [9, 10, 14, 13],
            [0, 1, 5, 4], [4, 5, 9, 8], [8, 9, 13, 12]
        ]

        face_centers = [[], [], []]

        for face in face_indices:
            face_center = (points_left[face[0]] + points_left[face[1]] +
                        points_left[face[2]] + points_left[face[3]]) / 4
            face_centers[0].append(face_center)


        for face in face_indices:
            face_center = (points_right[face[0]] + points_right[face[1]] +
                        points_right[face[2]] + points_right[face[3]]) / 4
            face_centers[1].append(face_center)


        for face in face_indices:
            face_center = (points_top[face[0]] + points_top[face[1]] +
                        points_top[face[2]] + points_top[face[3]]) / 4
            face_centers[2].append(face_center)


        # 12. Extract face colors
        reconstructed_faces = []
        faces_names = ["Left", "Right", "Top"]

        for f in range(3):
            reconstructed_face = np.zeros((3, 3, 3), dtype=np.uint8)
            for i in range(9):
                x, y = face_centers[f][i]
                x, y = int(x), int(y)
                w = center_sampling_width
                mean_color = img[y-w//2:y+w//2, x-w//2:x +
                                w//2].mean(axis=(0, 1)).astype(np.uint8)
                reconstructed_face[i//3, i % 3, :] = mean_color

            reconstructed_faces.append(reconstructed_face)
            
        # Fix face orientations
        # Right face
        reconstructed_faces[1] = np.flip(reconstructed_faces[1], axis=1)
        # Top face
        reconstructed_faces[2] = np.flip(reconstructed_faces[2], axis=1)
        reconstructed_faces[2] = np.flip(reconstructed_faces[2], axis=0)

        return reconstructed_faces
#endregion


# Main
colors = [
    (245, 0, 87), #rgb(245, 0, 87)
    (0, 230, 118), #rgb(0, 230, 118)
    (25, 118, 210), #rgb(25, 118, 210)
    (245, 124, 0), #rgb(245, 124, 0)
    (124, 77, 255) #rgb(124, 77, 255)
]

colors_01 = [
    (245/255, 0/255, 87/255), #rgb(245, 0, 87)
    (0/255, 230/255, 118/255), #rgb(0, 230, 118)
    (25/255, 118/255, 210/255), #rgb(25, 118, 210)
    (245/255, 124/255, 0/255), #rgb(245, 124, 0)
    (124/255, 77/255, 255/255) #rgb(124, 77, 255)
]

img = rgb(cv2.imread("input/rubiks1_15deg.jpg"))

def call_code(img):
    # fitted_ms_all, fitted_bs_all, scalers_all, points_left, points_right, points_top, reconstructed_faces = extract_faces(
    reconstructed_faces = extract_faces(
        img, 
        kernel_size=7,
        canny_low=0,
        canny_high=75,
        min_line_length=40,
        max_line_gap=20,
        center_sampling_width = 40,
        colors=colors,
        colors_01=colors_01,
        n_clusters=7,
        debug=False,
        debug_time=False)
    return reconstructed_faces

cap = cv2.VideoCapture(1)
i = 0
while True:
    ret, frame = cap.read()
    mask = np.zeros(frame.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    rect = (440,156,890,550)
    # cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), (255,0,0))
    # cv2.imshow("frame", frame)
    cv2.grabCut(frame,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    frame = frame*mask2[:,:,np.newaxis]
    cv2.imshow("grabcut", frame)
    # reconstructed_faces = call_code(frame)
    # cv2.imshow("Camera feed", frame)
    # display_reconstructed_faces(reconstructed_faces)
    print(i)
    i += 1
    if cv2.waitKey(200) == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
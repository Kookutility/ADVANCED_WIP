import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import Canvas, PhotoImage
from PIL import Image, ImageTk
from scipy.spatial.transform import Rotation as R
from pathlib import Path
import launch_setting_gui
from helpers import mediapipeTo3dpose
import pickle
from abw_wip import EAWIPTechnique
import mediapipe as mp
import time
from collections import deque
from numpy.fft import fft

script_dir = os.path.dirname(os.path.abspath(__file__))
icon_path = os.path.join(script_dir, 'assets', 'icon', 'VRlogy_icon.ico')
OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path("assets/frame4")

class CalibrationBase(tk.Frame):
    specific_landmark_indices = [2, 3, 1, 4, 20, 17]

    def __init__(self, root, params, camera_thread, pose, mp_drawing, max_frames, *args, **kwargs):
        tk.Frame.__init__(self, root, *args, **kwargs)
        self.params = params
        self.camera_thread = camera_thread
        self.pose = pose
        self.mp_drawing = mp_drawing
        self.root = root
        self.max_frames = max_frames
        self.frame_count = 0
        self.left_heel_heights = deque(maxlen=max_frames)
        self.right_heel_heights = deque(maxlen=max_frames)
        self.left_heel_positions = deque(maxlen=90)
        self.right_heel_positions = deque(maxlen=90)
        self.landmark_positions = [deque(maxlen=30) for _ in self.specific_landmark_indices]
        self.root.wm_iconbitmap(icon_path)
        self.setup_gui()

    def setup_gui(self):
        def relative_to_assets(path: str) -> Path:
            return ASSETS_PATH / Path(path)
        self.root.geometry("530x661+100+100")
        self.root.configure(bg="#FFFFFF")
        self.canvas = Canvas(self.root, bg="#FFFFFF", height=661, width=530, bd=0, highlightthickness=0, relief="ridge")
        self.canvas.place(x=0, y=0)
        self.image_image_1 = PhotoImage(file=relative_to_assets("image_1.png"))
        self.canvas.create_image(265.0, 330.5, image=self.image_image_1)
        self.canvas_video = self.canvas.create_image(265.0, 280.0)
        self.state_label = self.canvas.create_text(50, 50, text=f"{self.root.title()} 진행 중", fill="black", font=('Arial', 12), anchor="nw")

    def process_image(self, img):
        param = pickle.load(open("params.p", "rb"))
        img = cv2.resize(img, (param.get("camera_width"), param.get("camera_height")))
        if self.params.rotate_image is not None:
            img = cv2.rotate(img, self.params.rotate_image)
        if self.params.mirror:
            img = cv2.flip(img, 1)
        if max(img.shape) > self.params.maximgsize:
            ratio = max(img.shape) / self.params.maximgsize
            img = cv2.resize(img, (int(img.shape[1] / ratio), int(img.shape[0] / ratio)))
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def process_pose(self, results):
        # 2D 랜드마크로 발목 좌표 계산
        mp_pose = mp.solutions.pose.PoseLandmark
        left_heel_2d = results.pose_landmarks.landmark[mp_pose.LEFT_HEEL]
        right_heel_2d = results.pose_landmarks.landmark[mp_pose.RIGHT_HEEL]

        # Y값: 2D 좌표 기반 (정규화된 y를 미터 단위로 변환)
        y_scale = 1.0  # 카메라 높이 1m 가정
        left_heel_height = -0.5 + (1.0 - left_heel_2d.y) * y_scale
        right_heel_height = -0.5 + (1.0 - right_heel_2d.y) * y_scale

        # X값: 2D 좌표 기반 (정규화된 x를 미터 단위로 변환)
        x_scale = 1.0  # 카메라 FOV 기반
        left_heel_x = (left_heel_2d.x - 0.5) * x_scale
        right_heel_x = (right_heel_2d.x - 0.5) * x_scale

        # Z값: 고정값 (바닥 기준, 예: 0.5m)
        fixed_z = 0.5
        left_heel_pos = [left_heel_x, left_heel_height, fixed_z]
        right_heel_pos = [right_heel_x, right_heel_height, fixed_z]

        self.left_heel_positions.append(left_heel_pos)
        self.right_heel_positions.append(right_heel_pos)
        self.left_heel_40frame.append(left_heel_pos)
        self.right_heel_40frame.append(right_heel_pos)

        if self.frame_count % 3 == 0:
            print(f"Frame {self.frame_count}: Left Heel [X: {left_heel_pos[0]:.2f}, Y: {left_heel_pos[1]:.2f}, Z: {left_heel_pos[2]:.2f}], "
                  f"Right Heel [X: {right_heel_pos[0]:.2f}, Y: {right_heel_pos[1]:.2f}, Z: {right_heel_pos[2]:.2f}]")

        return left_heel_height, right_heel_height

    def update_video_feed(self):
        raise NotImplementedError

class Calibration1Window(CalibrationBase):
    def __init__(self, root, params, camera_thread, pose, mp_drawing, *args, **kwargs):
        super().__init__(root, params, camera_thread, pose, mp_drawing, max_frames=480, *args, **kwargs)
        self.root.title("1차 캘리브레이션")
        self.time_stamps = deque(maxlen=480)
        self.crossings_left = deque(maxlen=10)
        self.crossings_right = deque(maxlen=10)
        self.left_crossing_threshold = None
        self.right_crossing_threshold = None
        self.left_height_movement = 0.0
        self.right_height_movement = 0.0
        self.left_height_movements = []
        self.right_height_movements = []
        self.left_heel_40frame = deque(maxlen=40)
        self.right_heel_40frame = deque(maxlen=40)
        self.prev_left_heel_height = None
        self.prev_right_heel_height = None
        self.last_crossing_time_left = None
        self.last_crossing_time_right = None
        self.last_crossing_frame_left = None
        self.last_crossing_frame_right = None
        self.calibration_complete = False
        self.left_calib_frequency = 1.2
        self.right_calib_frequency = 1.3
        self.left_intervals = []
        self.right_intervals = []
        self.last_crossing_side = None
        self.update_video_feed()

    def update_video_feed(self):
        if self.frame_count >= self.max_frames:
            self.finish_calibration()
            self.calibration_complete = True
            return
        if not self.camera_thread.image_ready:
            self.root.after(10, self.update_video_feed)
            return

        img = self.process_image(self.camera_thread.image_from_thread.copy())
        self.camera_thread.image_ready = False
        results = self.pose.process(img)

        if results.pose_landmarks:
            left_heel_height, right_heel_height = self.process_pose(results)
            self.left_heel_heights.append(left_heel_height)
            self.right_heel_heights.append(right_heel_height)
            current_time = time.time()
            self.time_stamps.append(current_time)

            if not self.calibration_complete and self.frame_count > 90:
                left_y = np.array(list(self.left_heel_heights))
                right_y = np.array(list(self.right_heel_heights))
                self.left_crossing_threshold = np.mean(left_y) + np.std(left_y) if len(left_y) > 0 else 0.0
                self.right_crossing_threshold = np.mean(right_y) + np.std(right_y) if len(right_y) > 0 else 0.0

                if self.frame_count % 30 == 0:
                    print(f"Frame {self.frame_count}: Left Threshold: {self.left_crossing_threshold:.2f}, Right Threshold: {self.right_crossing_threshold:.2f}")
                    print(f"Debug: avg_left={np.mean(left_y):.2f}, std_left={np.std(left_y):.2f}, avg_right={np.mean(right_y):.2f}, std_right={np.std(right_y):.2f}")

                left_cross = (self.prev_left_heel_height < self.left_crossing_threshold <= left_heel_height) if self.prev_left_heel_height is not None else False
                right_cross = (self.prev_right_heel_height < self.right_crossing_threshold <= right_heel_height) if self.prev_right_heel_height is not None else False
                
                if left_cross:
                    if self.last_crossing_time_left is not None:
                        interval = current_time - self.last_crossing_time_left
                        frame_interval = self.frame_count - self.last_crossing_frame_left
                        if self.last_crossing_side == 'right' or frame_interval >= 10:
                            self.crossings_left.append((current_time, self.frame_count))
                            self.left_intervals.append(interval)
                            start_idx = max(0, len(self.left_heel_heights) - (self.frame_count - self.last_crossing_frame_left))
                            end_idx = len(self.left_heel_heights) - 1
                            between = list(self.left_heel_heights)[start_idx:end_idx + 1]
                            if between:
                                height_diff = max(between) - min(between)
                                self.left_height_movements.append(height_diff)
                                print(f"Frame {self.frame_count}: Left Height Movement: {height_diff:.4f}")
                            self.last_crossing_time_left = current_time
                            self.last_crossing_frame_left = self.frame_count
                            self.last_crossing_side = 'left'
                            stride_freq_left = min(1.0 / interval if interval > 0 else 0, 5.0)
                            print(f"Frame {self.frame_count}: Left Crossing Detected (Upward), Interval: {interval:.3f}s, Stride Frequency: {stride_freq_left:.2f} Hz")
                    else:
                        self.last_crossing_time_left = current_time
                        self.last_crossing_frame_left = self.frame_count
                        self.last_crossing_side = 'left'
                        self.crossings_left.append((current_time, self.frame_count))

                if right_cross:
                    if self.last_crossing_time_right is not None:
                        interval = current_time - self.last_crossing_time_right
                        frame_interval = self.frame_count - self.last_crossing_frame_right
                        if self.last_crossing_side == 'left' or frame_interval >= 10:
                            self.crossings_right.append((current_time, self.frame_count))
                            self.right_intervals.append(interval)
                            start_idx = max(0, len(self.right_heel_heights) - (self.frame_count - self.last_crossing_frame_right))
                            end_idx = len(self.right_heel_heights) - 1
                            between = list(self.right_heel_heights)[start_idx:end_idx + 1]
                            if between:
                                height_diff = max(between) - min(between)
                                self.right_height_movements.append(height_diff)
                                print(f"Frame {self.frame_count}: Right Height Movement: {height_diff:.4f}")
                            self.last_crossing_time_right = current_time
                            self.last_crossing_frame_right = self.frame_count
                            self.last_crossing_side = 'right'
                            stride_freq_right = min(1.0 / interval if interval > 0 else 0, 5.0)
                            print(f"Frame {self.frame_count}: Right Crossing Detected (Upward), Interval: {interval:.3f}s, Stride Frequency: {stride_freq_right:.2f} Hz")
                    else:
                        self.last_crossing_time_right = current_time
                        self.last_crossing_frame_right = self.frame_count
                        self.last_crossing_side = 'right'
                        self.crossings_right.append((current_time, self.frame_count))

            self.prev_left_heel_height = left_heel_height
            self.prev_right_heel_height = right_heel_height
            self.mp_drawing.draw_landmarks(img, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)

        self.display_image(img)
        self.frame_count += 1
        self.root.after(10, self.update_video_feed)

    def finish_calibration(self):
        if len(self.left_heel_heights) < 90 or len(self.crossings_left) < 2 or len(self.crossings_right) < 2:
            self.left_crossing_threshold = -0.44
            self.right_crossing_threshold = -0.42
            self.left_height_movement = 0.10
            self.right_height_movement = 0.12
            self.left_calib_frequency = 1.2
            self.right_calib_frequency = 1.3
            print("데이터 부족, 기본값 적용")
        else:
            avg_crossing_left = np.mean(self.left_intervals) if self.left_intervals else 1.0
            avg_crossing_right = np.mean(self.right_intervals) if self.right_intervals else 1.0
            self.left_calib_frequency = min(1.0 / avg_crossing_left if avg_crossing_left > 0 else 1.2, 4.5)
            self.right_calib_frequency = min(1.0 / avg_crossing_right if avg_crossing_right > 0 else 1.3, 4.5)

            self.left_height_movement = np.mean(self.left_height_movements) if self.left_height_movements else 0.10
            self.right_height_movement = np.mean(self.right_height_movements) if self.right_height_movements else 0.12

            left_x = np.array([pos[0] for pos in self.left_heel_40frame])
            left_y = np.array([pos[1] for pos in self.left_heel_40frame])
            left_z = np.array([pos[2] for pos in self.left_heel_40frame])
            right_x = np.array([pos[0] for pos in self.right_heel_40frame])
            right_y = np.array([pos[1] for pos in self.right_heel_40frame])
            right_z = np.array([pos[2] for pos in self.right_heel_40frame])

            delta_x_left = np.max(left_x) - np.min(left_x)
            delta_y_left = np.max(left_y) - np.min(left_y)
            delta_z_left = np.max(left_z) - np.min(left_z)
            delta_x_right = np.max(right_x) - np.min(right_x)
            delta_y_right = np.max(right_y) - np.min(right_y)
            delta_z_right = np.max(right_z) - np.min(right_z)

            score_left = 1 * delta_x_left + 1 * delta_y_left + 1 * delta_z_left
            score_right = 1 * delta_x_right + 1 * delta_y_right + 1 * delta_z_right


            print(f"Calibration Done: Left Threshold: {self.left_crossing_threshold:.2f}, Right Threshold: {self.right_crossing_threshold:.2f}, "
                  f"Left Height Movement: {self.left_height_movement:.2f}, Right Height Movement: {self.right_height_movement:.2f}, "
                  f"Left Calib Frequency: {self.left_calib_frequency:.2f} Hz, Right Calib Frequency: {self.right_calib_frequency:.2f} Hz")

        self.root.destroy()

    def get_results(self):
        return (self.left_crossing_threshold, self.right_crossing_threshold, 
                self.left_calib_frequency, self.right_calib_frequency, self.left_height_movement, self.right_height_movement)

    def display_image(self, img):
        imgtk = ImageTk.PhotoImage(image=Image.fromarray(img))
        self.canvas.itemconfig(self.canvas_video, image=imgtk)
        self.canvas.image = imgtk

class InferenceWindow(tk.Frame):
    def __init__(self, root, params, camera_thread, backend, pose, mp_drawing, eawip_technique, ground_level_3d, *args, **kwargs):
        tk.Frame.__init__(self, root, *args, **kwargs)
        self.params = params
        self.camera_thread = camera_thread
        self.backend = backend
        self.pose = pose
        self.mp_drawing = mp_drawing
        self.walking_detector = eawip_technique
        self.root = root
        self.frame_count = 0
        self.left_heel_positions = deque(maxlen=90)
        self.right_heel_positions = deque(maxlen=90)
        self.left_heel_40frame = deque(maxlen=40)  # 추가
        self.right_heel_40frame = deque(maxlen=40)  # 추가
        self.root.wm_iconbitmap(icon_path)
        self.root.title("VRlogy")
        self.setup_gui()
        self.update_video_feed()

    def setup_gui(self):
        def relative_to_assets(path: str) -> Path:
            return ASSETS_PATH / Path(path)

        def on_button_click(button_id):
            if button_id == 1:
                self.root.destroy()
                launch_setting_gui.make_gui(self.params)
            elif button_id == 3:
                self.params.change_mirror(not self.params.mirror)

        self.root.geometry("530x661+100+100")
        self.root.configure(bg="#FFFFFF")
        self.canvas = Canvas(self.root, bg="#FFFFFF", height=661, width=530, bd=0, highlightthickness=0, relief="ridge")
        self.canvas.place(x=0, y=0)
        self.image_image_1 = PhotoImage(file=relative_to_assets("image_1.png"))
        self.canvas.create_image(265, 380.0, image=self.image_image_1)
        self.button_image_1 = PhotoImage(file=relative_to_assets("button_1.png"))
        self.button_image_3 = PhotoImage(file=relative_to_assets("button_3.png"))
        button_1 = self.canvas.create_image(265, 611, image=self.button_image_1, anchor="center")
        self.canvas.tag_bind(button_1, "<Button-1>", lambda e: on_button_click(1))
        button_3 = self.canvas.create_image(68, 548, image=self.button_image_3, anchor="center")
        self.canvas.tag_bind(button_3, "<Button-1>", lambda e: on_button_click(3))
        self.canvas_video = self.canvas.create_image(265.0, 280.0)
        self.state_label = self.canvas.create_text(50, 50, text="Speed: 0.00 m/s", fill="black", font=('Arial', 24), anchor="nw")

    def process_image(self, img):
        param = pickle.load(open("params.p", "rb"))
        img = cv2.resize(img, (param.get("camera_width"), param.get("camera_height")))
        if self.params.rotate_image is not None:
            img = cv2.rotate(img, self.params.rotate_image)
        if self.params.mirror:
            img = cv2.flip(img, 1)
        if max(img.shape) > self.params.maximgsize:
            ratio = max(img.shape) / self.params.maximgsize
            img = cv2.resize(img, (int(img.shape[1] / ratio), int(img.shape[0] / ratio)))
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def update_video_feed(self):
        if not self.camera_thread.image_ready:
            self.root.after(10, self.update_video_feed)
            return

        img = self.process_image(self.camera_thread.image_from_thread.copy())
        self.camera_thread.image_ready = False
        results = self.pose.process(img)
        
        speed, warning = 0.0, False
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            mp_pose = mp.solutions.pose.PoseLandmark
            left_heel_vis = landmarks[mp_pose.LEFT_HEEL].visibility
            right_heel_vis = landmarks[mp_pose.RIGHT_HEEL].visibility
            left_heel_visible = left_heel_vis >= 0.75
            right_heel_visible = right_heel_vis >= 0.75

            # 2D 기반 좌표 계산
            left_heel_height, right_heel_height = CalibrationBase.process_pose(self, results)
            left_heel_pos = self.left_heel_positions[-1] if self.left_heel_positions else [0, left_heel_height, 0.5]
            right_heel_pos = self.right_heel_positions[-1] if self.right_heel_positions else [0, right_heel_height, 0.5]

            # pose3d는 다른 랜드마크(손 등)에 필요
            pose3d = mediapipeTo3dpose(results.pose_world_landmarks.landmark)
            pose3d[:, 0], pose3d[:, 1] = -pose3d[:, 0], -pose3d[:, 1]
            for j in range(pose3d.shape[0]):
                pose3d[j] = self.params.global_rot_z.apply(pose3d[j])
                pose3d[j] = self.params.global_rot_x.apply(pose3d[j])
                pose3d[j] = self.params.global_rot_y.apply(pose3d[j])
            # 발목 좌표는 2D 기반으로 오버라이드
            pose3d[0] = left_heel_pos
            pose3d[5] = right_heel_pos

            speed, _ = self.walking_detector.update( 
                pose3d, left_heel_visible, right_heel_visible,
                left_visibility=left_heel_vis, right_visibility=right_heel_vis,
                visibilities=[landmarks[i].visibility for i in range(len(landmarks))],
                warning=warning, left_heel_height=left_heel_height,
                right_heel_height=right_heel_height)
            self.canvas.itemconfig(self.state_label, text=f"Speed: {speed:.2f} m/s")
            self.mp_drawing.draw_landmarks(img, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)

        imgtk = ImageTk.PhotoImage(image=Image.fromarray(img))
        self.canvas.itemconfig(self.canvas_video, image=imgtk)
        if self.frame_count % 10 == 0:
            print(f"left_heel_visible = {left_heel_visible}, "
                  f"right_heel_visible = {right_heel_visible}")
        self.canvas.image = imgtk
        self.frame_count += 1
        self.root.after(10, self.update_video_feed)

def run_calibration_and_tracking(root, params, camera_thread, backend, pose, mp_drawing):
    print("Starting Calibration 1")
    calib1_root = tk.Tk()
    calib1_window = Calibration1Window(calib1_root, params, camera_thread, pose, mp_drawing)
    calib1_root.mainloop()
    
    left_crossing_threshold, right_crossing_threshold, left_calib_frequency, right_calib_frequency, left_calib_height, right_calib_height = calib1_window.get_results()
    print(f"Calibration Done: Left Threshold: {left_crossing_threshold:.2f}, Right Threshold: {right_crossing_threshold:.2f}, "
          f"Left Calib Frequency: {left_calib_frequency:.2f}, Right Calib Frequency: {right_calib_frequency:.2f}, "
          f"Left Calib Height: {left_calib_height:.2f}, Right Calib Height: {right_calib_height:.2f}")

    eawip_technique = EAWIPTechnique()
    eawip_technique.set_calibration_results(left_crossing_threshold, right_crossing_threshold, 
                                           left_calib_frequency, right_calib_frequency, left_calib_height, right_calib_height)
    print("캘리브레이션 적용 완료")

    inference_root = tk.Tk()
    InferenceWindow(inference_root, params, camera_thread, backend, pose, mp_drawing, eawip_technique, (0.0, 0.0)).pack(side="top", fill="both", expand=True)
    inference_root.mainloop()
    return eawip_technique
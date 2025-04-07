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
    specific_landmark_indices = [2, 3, 1, 4, 20, 17]  # 발뒤꿈치: 20 (LEFT_HEEL), 17 (RIGHT_HEEL)

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
        self.left_heel_positions = deque(maxlen=90)  # [X, Y, Z]
        self.right_heel_positions = deque(maxlen=90)  # [X, Y, Z]
        self.hip_heights = deque(maxlen=90)
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

    def process_pose(self, pose3d):
        left_heel_pos = pose3d[0]  # LEFT_ANKLE
        right_heel_pos = pose3d[5]  # RIGHT_ANKLE
        left_hip_pos = pose3d[2]    # LEFT_HIP
        right_hip_pos = pose3d[3]   # RIGHT_HIP

        self.left_heel_positions.append(left_heel_pos)
        self.right_heel_positions.append(right_heel_pos)

        left_heel_height = left_heel_pos[1]
        right_heel_height = right_heel_pos[1]
        avg_hip_height = (left_hip_pos[1] + right_hip_pos[1]) / 2

        if self.frame_count % 3 == 0:
            print(f"Frame {self.frame_count}: Left Heel [X: {left_heel_pos[0]:.2f}, Y: {left_heel_pos[1]:.2f}, Z: {left_heel_pos[2]:.2f}], "
                  f"Right Heel [X: {right_heel_pos[0]:.2f}, Y: {right_heel_pos[1]:.2f}, Z: {right_heel_pos[2]:.2f}]")

        return left_heel_height, right_heel_height, avg_hip_height

    def update_video_feed(self):
        raise NotImplementedError

class Calibration1Window(CalibrationBase):
    def __init__(self, root, params, camera_thread, pose, mp_drawing, *args, **kwargs):
        super().__init__(root, params, camera_thread, pose, mp_drawing, max_frames=480, *args, **kwargs)
        self.root.title("1차 캘리브레이션")
        self.time_stamps = deque(maxlen=480)
        self.crossings_left = []  # (시간, 인터벌)
        self.crossings_right = []
        self.left_crossing_threshold = None
        self.right_crossing_threshold = None
        self.accel_factor = 1.0
        self.base_noise_score = 0.38  # 기본값
        self.left_height_movement = 0.0
        self.right_height_movement = 0.0
        self.left_heel_40frame = deque(maxlen=40)  # 노이즈 기준용
        self.right_heel_40frame = deque(maxlen=40)
        self.prev_left_heel_height = None
        self.prev_right_heel_height = None
        self.last_crossing_time_left = None
        self.last_crossing_time_right = None
        self.calibration_complete = False  # 캘리브레이션 완료 여부 플래그 추가
        self.update_video_feed()

    def update_video_feed(self):
        if self.frame_count >= self.max_frames:
            self.finish_calibration()
            self.calibration_complete = True  # 캘리브레이션 완료 표시
            return
        if not self.camera_thread.image_ready:
            self.root.after(10, self.update_video_feed)
            return

        img = self.process_image(self.camera_thread.image_from_thread.copy())
        self.camera_thread.image_ready = False
        results = self.pose.process(img)

        if results.pose_landmarks:
            pose3d = mediapipeTo3dpose(results.pose_world_landmarks.landmark)
            pose3d[:, 0], pose3d[:, 1] = -pose3d[:, 0], -pose3d[:, 1]
            for j in range(pose3d.shape[0]):
                pose3d[j] = self.params.global_rot_z.apply(pose3d[j])
                pose3d[j] = self.params.global_rot_x.apply(pose3d[j])
                pose3d[j] = self.params.global_rot_y.apply(pose3d[j])

            left_heel_height, right_heel_height, avg_hip_height = self.process_pose(pose3d)
            self.left_heel_heights.append(left_heel_height)
            self.right_heel_heights.append(right_heel_height)
            self.left_heel_positions.append(pose3d[0])
            self.right_heel_positions.append(pose3d[5])
            self.left_heel_40frame.append(pose3d[0])
            self.right_heel_40frame.append(pose3d[5])
            current_time = time.time()
            self.time_stamps.append(current_time)

            # Frame 90~240: 임계값 계산
            if self.frame_count == 240:
                left_y = np.array(list(self.left_heel_heights))
                right_y = np.array(list(self.right_heel_heights))
                avg_left = np.mean(left_y)
                avg_right = np.mean(right_y)
                std_left = np.std(left_y)
                std_right = np.std(right_y)
                self.left_crossing_threshold = avg_left + std_left
                self.right_crossing_threshold = avg_right + std_right
                print(f"Frame 240: Left Threshold: {self.left_crossing_threshold:.2f}, Right Threshold: {self.right_crossing_threshold:.2f}")
                print(f"Debug: avg_left={avg_left:.2f}, std_left={std_left:.2f}, avg_right={avg_right:.2f}, std_right={std_right:.2f}")

            # Frame 240~480: 교차 주기 감지 및 로그 출력 (캘리브레이션 중에만)
            if not self.calibration_complete and self.frame_count > 240 and self.left_crossing_threshold is not None and self.right_crossing_threshold is not None:
                left_cross = (self.prev_left_heel_height < self.left_crossing_threshold <= left_heel_height or 
                              self.prev_left_heel_height >= self.left_crossing_threshold > left_heel_height)
                right_cross = (self.prev_right_heel_height < self.right_crossing_threshold <= right_heel_height or 
                               self.prev_right_heel_height >= self.right_crossing_threshold > right_heel_height)
                
                if left_cross and self.last_crossing_time_left is not None:
                    interval = current_time - self.last_crossing_time_left
                    self.crossings_left.append((current_time, interval))
                    self.last_crossing_time_left = current_time
                    stride_freq_left = 1 / interval if interval > 0 else 0
                    print(f"Frame {self.frame_count}: Left Crossing Detected, Interval: {interval:.3f}s, Stride Frequency: {stride_freq_left:.2f} Hz")
                elif left_cross:
                    self.last_crossing_time_left = current_time

                if right_cross and self.last_crossing_time_right is not None:
                    interval = current_time - self.last_crossing_time_right
                    self.crossings_right.append((current_time, interval))
                    self.last_crossing_time_right = current_time
                    stride_freq_right = 1 / interval if interval > 0 else 0
                    print(f"Frame {self.frame_count}: Right Crossing Detected, Interval: {interval:.3f}s, Stride Frequency: {stride_freq_right:.2f} Hz")
                elif right_cross:
                    self.last_crossing_time_right = current_time

            self.prev_left_heel_height = left_heel_height
            self.prev_right_heel_height = right_heel_height
            self.mp_drawing.draw_landmarks(img, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)

        self.display_image(img)
        self.frame_count += 1
        self.root.after(10, self.update_video_feed)

    def finish_calibration(self):
        if len(self.left_heel_heights) < 240 or len(self.crossings_left) < 1 or len(self.crossings_right) < 1:
            self.left_crossing_threshold = -0.44
            self.right_crossing_threshold = -0.42
            self.accel_factor = 1.0  # 기존 기본값
            self.base_noise_score = 0.38
            self.left_height_movement = 0.10
            self.right_height_movement = 0.12
            print("데이터 부족, 기본값 적용")
        else:
            # 교차 주기 계산 (독립적으로)
            intervals_left = [interval for _, interval in self.crossings_left]
            intervals_right = [interval for _, interval in self.crossings_right]
            trimmed_left = np.percentile(intervals_left, [10, 90])
            trimmed_right = np.percentile(intervals_right, [10, 90])
            avg_crossing_left = np.mean([i for i in intervals_left if trimmed_left[0] <= i <= trimmed_left[1]])
            avg_crossing_right = np.mean([i for i in intervals_right if trimmed_right[0] <= i <= trimmed_right[1]])
            self.left_calib_frequency = 1 / avg_crossing_left if avg_crossing_left > 0 else 1.2
            self.right_calib_frequency = 1 / avg_crossing_right if avg_crossing_right > 0 else 1.3
            # self.calib_frequency는 제거 (독립 주기 사용)

            # 높이 이동값 계산
            left_y = np.array(list(self.left_heel_heights)[-240:])
            right_y = np.array(list(self.right_heel_heights)[-240:])
            self.left_height_movement = np.mean(np.abs(np.diff(left_y)))
            self.right_height_movement = np.mean(np.abs(np.diff(right_y)))
            print(f"Debug: left_y range={np.max(left_y):.2f} - {np.min(left_y):.2f}, height_movement={self.left_height_movement:.4f}")
            print(f"Debug: right_y range={np.max(right_y):.2f} - {np.min(right_y):.2f}, height_movement={self.right_height_movement:.4f}")

            # 가속도 계수 계산 (독립적으로)
            target_speed = 1.399  # 목표 속도
            left_calib_speed = self.left_height_movement * self.left_calib_frequency
            right_calib_speed = self.right_height_movement * self.right_calib_frequency
            self.accel_factor_left = target_speed / left_calib_speed if left_calib_speed > 0 else 1.0
            self.accel_factor_right = target_speed / right_calib_speed if right_calib_speed > 0 else 1.0

            # 노이즈 스코어 계산 (기존 유지)
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
            self.base_noise_score = (score_left + score_right) / 2  # 평균 유지

            # 결과 출력
            print(f"Calibration Done: Left Threshold: {self.left_crossing_threshold:.2f}, Right Threshold: {self.right_crossing_threshold:.2f}, "
                  f"Accel Factor Left: {self.accel_factor_left:.2f}, Accel Factor Right: {self.accel_factor_right:.2f}, "
                  f"Base Noise Score: {self.base_noise_score:.2f}, "
                  f"Left Height Movement: {self.left_height_movement:.2f}, Right Height Movement: {self.right_height_movement:.2f}")
            print(f"Left Calibrated Stride Frequency: {self.left_calib_frequency:.2f} Hz, Right Calibrated Stride Frequency: {self.right_calib_frequency:.2f} Hz")
            print(f"1.399 Mapping: Left = accel_factor_left ({self.accel_factor_left:.2f}) * left_height_movement ({self.left_height_movement:.4f}) * left_calib_frequency ({self.left_calib_frequency:.2f}) ≈ 1.399")
            print(f"1.399 Mapping: Right = accel_factor_right ({self.accel_factor_right:.2f}) * right_height_movement ({self.right_height_movement:.4f}) * right_calib_frequency ({self.right_calib_frequency:.2f}) ≈ 1.399")

        self.root.destroy()

    def get_results(self):
        # 수정: accel_factor 대신 left/right 독립 값 반환
        return (self.left_crossing_threshold, self.right_crossing_threshold, self.accel_factor_left, self.accel_factor_right, self.base_noise_score, self.left_calib_frequency, self.right_calib_frequency)
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
            left_heel_visible = left_heel_vis >= 0.8
            right_heel_visible = right_heel_vis >= 0.8

            pose3d = mediapipeTo3dpose(results.pose_world_landmarks.landmark)
            pose3d[:, 0], pose3d[:, 1] = -pose3d[:, 0], -pose3d[:, 1]
            for j in range(pose3d.shape[0]):
                pose3d[j] = self.params.global_rot_z.apply(pose3d[j])
                pose3d[j] = self.params.global_rot_x.apply(pose3d[j])
                pose3d[j] = self.params.global_rot_y.apply(pose3d[j])

            left_heel_height, right_heel_height, avg_hip_height = CalibrationBase.process_pose(self, pose3d)

            speed, _ = self.walking_detector.update(pose3d, left_heel_visible, right_heel_visible,
                                                    left_visibility=left_heel_vis, right_visibility=right_heel_vis,
                                                    visibilities=[landmarks[i].visibility for i in range(len(landmarks))],
                                                    warning=warning, left_heel_height=left_heel_height,
                                                    right_heel_height=right_heel_height, avg_hip_height=avg_hip_height)
            self.canvas.itemconfig(self.state_label, text=f"Speed: {speed:.2f} m/s")
            self.mp_drawing.draw_landmarks(img, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)

        imgtk = ImageTk.PhotoImage(image=Image.fromarray(img))
        self.canvas.itemconfig(self.canvas_video, image=imgtk)
        self.canvas.image = imgtk
        self.frame_count += 1
        self.root.after(10, self.update_video_feed)

def run_calibration_and_tracking(root, params, camera_thread, backend, pose, mp_drawing):
    print("Starting Calibration 1")
    calib1_root = tk.Tk()
    calib1_window = Calibration1Window(calib1_root, params, camera_thread, pose, mp_drawing)
    calib1_root.mainloop()
    
    # 7개 값 받기
    left_crossing_threshold, right_crossing_threshold, accel_factor_left, accel_factor_right, base_noise_score, left_calib_frequency, right_calib_frequency = calib1_window.get_results()
    if left_crossing_threshold is None or right_crossing_threshold is None or accel_factor_left is None or accel_factor_right is None:
        print("1차 캘리브레이션 실패")
        return None
    print(f"Calibration Done: Left Threshold: {left_crossing_threshold:.2f}, Right Threshold: {right_crossing_threshold:.2f}, "
          f"Accel Factor Left: {accel_factor_left:.2f}, Accel Factor Right: {accel_factor_right:.2f}, "
          f"Left Calib Frequency: {left_calib_frequency:.2f}, Right Calib Frequency: {right_calib_frequency:.2f}")

    eawip_technique = EAWIPTechnique()
    eawip_technique.set_calibration_results(left_crossing_threshold, right_crossing_threshold, accel_factor_left, accel_factor_right, base_noise_score, left_calib_frequency, right_calib_frequency)
    print("캘리브레이션 적용 완료")

    inference_root = tk.Tk()
    InferenceWindow(inference_root, params, camera_thread, backend, pose, mp_drawing, eawip_technique, (0.0, 0.0)).pack(side="top", fill="both", expand=True)
    inference_root.mainloop()
    return eawip_technique
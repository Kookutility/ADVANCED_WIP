import time
import numpy as np
import socket
from collections import deque
from numpy.fft import fft

class EAWIPTechnique:
    def __init__(self, udp_ip="127.0.0.1", udp_port=5005):
        self.max_speed_walking = 6.3
        self.max_speed_running = 13.23  # 사용 안 함, 참고용으로 유지
        self.window_size = 90
        self.current_speed = 0.0
        self.target_speed = 0.0
        
        self.left_heel_heights = deque(maxlen=90)
        self.right_heel_heights = deque(maxlen=90)
        self.left_heel_smoothing = deque(maxlen=5)
        self.right_heel_smoothing = deque(maxlen=5)
        self.time_stamps = deque(maxlen=90)
        self.crossings_left = deque(maxlen=30)
        self.crossings_right = deque(maxlen=30)
        self.left_heel_40frame = deque(maxlen=40)  # 노이즈 판단용
        self.right_heel_40frame = deque(maxlen=40)
        
        self.udp_ip = udp_ip
        self.udp_port = udp_port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.last_sent_speed = None
        self.speed_change_threshold = 0.01
        self.last_update_time = 0.0
        self.frame_count = 0
        
        self.left_crossing_threshold = 0.0
        self.right_crossing_threshold = 0.0
        self.accel_factor = 1.0
        self.base_noise_score = 0.38  # 캘리브레이션 기본값
        
        self.prev_left_heel_height = None
        self.prev_right_heel_height = None
        self.last_crossing_time_left = None
        self.last_crossing_time_right = None
        
        self.weber_fraction = 0.1
        self.last_applied_speed = 0.0
        
        # 수정된 부분: maxlen=10 -> maxlen=30
        self.left_heel_buffer = deque(maxlen=30)
        self.right_heel_buffer = deque(maxlen=30)
        
        self.last_motion_time = time.time()
        self.motion_timeout = 2.0

    def set_calibration_results(self, left_crossing_threshold, right_crossing_threshold, accel_factor, base_noise_score, calib_frequency):
        self.left_crossing_threshold = left_crossing_threshold
        self.right_crossing_threshold = right_crossing_threshold
        self.accel_factor = accel_factor
        self.base_noise_score = base_noise_score
        self.calib_frequency = calib_frequency
        print(f"Calibration Applied - Left Threshold: {self.left_crossing_threshold:.2f}, Right Threshold: {self.right_crossing_threshold:.2f}, "
              f"Accel Factor: {self.accel_factor:.2f}, Base Noise Score: {self.base_noise_score:.2f}, Calib Frequency: {self.calib_frequency:.2f}")
    def detect_frontal_speed(self):
        if len(self.left_heel_heights) < 30 or len(self.right_heel_heights) < 30:
            return 0.0, 0.0

        if len(self.left_heel_buffer) >= 30 and len(self.right_heel_buffer) >= 30:
            left_change = max(self.left_heel_buffer) - min(self.left_heel_buffer)
            right_change = max(self.right_heel_buffer) - min(self.right_heel_buffer)
            if left_change <= 0.02 and right_change <= 0.02:
                self.crossings_left.clear()
                self.crossings_right.clear()
                return 0.0, 0.0

        # 수정: np.diff 대신 max - min 사용
        recent_left_heights = np.array(list(self.left_heel_heights)[-30:])
        recent_right_heights = np.array(list(self.right_heel_heights)[-30:])
        height_movement_left = np.max(recent_left_heights) - np.min(recent_left_heights)
        height_movement_right = np.max(recent_right_heights) - np.min(recent_right_heights)

        crossings_left = [t for t in self.crossings_left if t > 0]
        crossings_right = [t for t in self.crossings_right if t > 0]
        avg_cross_left = np.mean(crossings_left) if crossings_left else 1.0
        avg_cross_right = np.mean(crossings_right) if crossings_right else 1.0
        
        stride_frequency = max(1.0 / avg_cross_left if avg_cross_left > 0 else 0, 1.0 / avg_cross_right if avg_cross_right > 0 else 0)
        crossing_factor_left = min(stride_frequency / self.calib_frequency, 1.5) if self.calib_frequency > 0 else 1.0
        crossing_factor_right = min(stride_frequency / self.calib_frequency, 1.5) if self.calib_frequency > 0 else 1.0

        speed_left = self.accel_factor * height_movement_left * crossing_factor_left
        speed_right = self.accel_factor * height_movement_right * crossing_factor_right
        target_speed = max(speed_left, speed_right)

        return target_speed, stride_frequency
    def calculate_vibration(self):
        return 0.0

    def is_running(self):
        return False

    def detect_sit_stand_noise(self):
        if len(self.left_heel_40frame) < 40 or len(self.right_heel_40frame) < 40:
            return False

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

        score_left = 1 * delta_x_left + 3 * delta_y_left + 1 * delta_z_left
        score_right = 1 * delta_x_right + 3 * delta_y_right + 1 * delta_z_right
        noise_threshold = self.base_noise_score * 3.0

        is_noise = score_left > noise_threshold or score_right > noise_threshold
        if is_noise:
            print(f"Debug: Left Score: {score_left:.2f}, Right Score: {score_right:.2f}, Threshold: {noise_threshold:.2f}")
        return is_noise

    def process_heel_data(self, pose3d, left_heel_visible, right_heel_visible, left_visibility=0.0, right_visibility=0.0, 
                          visibilities=None, left_heel_height=None, right_heel_height=None):
        if pose3d is None or len(pose3d) < 31:
            return 0.0, 0, 0

        self.frame_count += 1
        current_time = time.time()
        self.time_stamps.append(current_time)

        left_heel_pos = pose3d[0]
        right_heel_pos = pose3d[5]
        self.left_heel_40frame.append(left_heel_pos)
        self.right_heel_40frame.append(right_heel_pos)

        left_heel_height = left_heel_pos[1]
        right_heel_height = right_heel_pos[1]
        self.left_heel_heights.append(left_heel_height)
        self.right_heel_heights.append(right_heel_height)
        self.left_heel_buffer.append(left_heel_height)
        self.right_heel_buffer.append(right_heel_height)

        if self.prev_left_heel_height is not None:
            left_cross = (self.prev_left_heel_height < self.left_crossing_threshold <= left_heel_height or 
                          self.prev_left_heel_height >= self.left_crossing_threshold > left_heel_height)
            if left_cross and self.last_crossing_time_left is not None:
                interval = current_time - self.last_crossing_time_left
                self.crossings_left.append(interval)
                self.last_crossing_time_left = current_time
            elif left_cross:
                self.last_crossing_time_left = current_time

        if self.prev_right_heel_height is not None:
            right_cross = (self.prev_right_heel_height < self.right_crossing_threshold <= right_heel_height or 
                           self.prev_right_heel_height >= self.right_crossing_threshold > right_heel_height)
            if right_cross and self.last_crossing_time_right is not None:
                interval = current_time - self.last_crossing_time_right
                self.crossings_right.append(interval)
                self.last_crossing_time_right = current_time
            elif right_cross:
                self.last_crossing_time_right = current_time

        self.prev_left_heel_height = left_heel_height
        self.prev_right_heel_height = right_heel_height

        if self.frame_count % 10 == 0:
            print(f"Frame {self.frame_count}: Left Heel: {left_heel_height:.2f}, Right Heel: {right_heel_height:.2f}")

        target_speed, stride_frequency = self.detect_frontal_speed()
        return target_speed, stride_frequency, 0

    def send_speed_to_unity(self, warning=False):
        speed_data = f"{self.current_speed:.4f}:{1 if warning else 0}"
        if self.last_sent_speed is None or abs(self.current_speed - self.last_sent_speed) > (self.last_sent_speed * self.weber_fraction):
            self.sock.sendto(speed_data.encode('utf-8'), (self.udp_ip, self.udp_port))
            self.last_sent_speed = self.current_speed
            print(f"Speed sent to Unity: {self.current_speed:.2f}")

    def update(self, pose3d, left_heel_visible, right_heel_visible, left_visibility=0.0, right_visibility=0.0, 
               visibilities=None, warning=False, speed_multiplier=1.0, left_heel_height=None, right_heel_height=None, 
               avg_hip_height=None):
        current_time = time.time()
        dt = current_time - self.last_update_time if self.last_update_time > 0 else 0.01
        self.last_update_time = current_time

        target_speed = 0.0
        stride_frequency = 0.0

        if pose3d is None or len(pose3d) < 31:
            self.current_speed = 0.0
            self.last_applied_speed = 0.0
            self.send_speed_to_unity(warning=warning)
            return self.current_speed, 0

        if self.detect_sit_stand_noise():
            self.current_speed = 0.0
            self.last_applied_speed = 0.0
            self.send_speed_to_unity(warning=warning)
            print(f"Frame {self.frame_count}: Sit/Stand noise detected, Speed forced to 0")
            return self.current_speed, 0

        if left_heel_visible or right_heel_visible:
            target_speed, stride_frequency, _ = self.process_heel_data(
                pose3d, left_heel_visible, right_heel_visible, left_visibility, right_visibility, visibilities,
                left_heel_height, right_heel_height)
            max_speed = self.max_speed_walking

            if stride_frequency > 0:
                self.last_motion_time = current_time
            if current_time - self.last_motion_time > self.motion_timeout:
                target_speed = 0.0

        speed_diff = abs(target_speed - self.last_applied_speed)
        weber_threshold = max(0.1, self.last_applied_speed * self.weber_fraction)
        if speed_diff >= weber_threshold or target_speed == 0.0:
            adjusted_speed = target_speed
            self.current_speed = 0.7 * self.current_speed + 0.3 * min(adjusted_speed, max_speed) if target_speed > 0 else 0.0
            self.last_applied_speed = target_speed

        if not (left_heel_visible or right_heel_visible):
            self.current_speed = self.target_speed = 0.0
            self.last_applied_speed = 0.0

        self.current_speed = max(0.0, self.current_speed * speed_multiplier)
        self.send_speed_to_unity(warning=warning)

        if self.frame_count % 10 == 0:
            print(f"Frame {self.frame_count}: Speed: {self.current_speed:.2f}, Target Speed: {target_speed:.2f}, "
                  f"Stride Frequency: {stride_frequency:.1f}")

        return self.current_speed, stride_frequency
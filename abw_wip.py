import time
import numpy as np
import socket
from collections import deque
from numpy.fft import fft

class EAWIPTechnique:
    def __init__(self, udp_ip="127.0.0.1", udp_port=5005):
        self.max_speed_walking = 7
        self.max_speed_running = 13.23
        self.window_size = 90
        self.current_speed = 0.0
        self.target_speed = 0.0
        
        self.left_heel_heights = deque(maxlen=180)
        self.right_heel_heights = deque(maxlen=180)
        self.left_heel_smoothing = deque(maxlen=5)
        self.right_heel_smoothing = deque(maxlen=5)
        self.time_stamps = deque(maxlen=180)
        self.crossings_left = deque(maxlen=10)
        self.right_crossings = deque(maxlen=10)
        self.left_heel_40frame = deque(maxlen=40)
        self.right_heel_40frame = deque(maxlen=40)
        
        self.noise_counter = 0
        self.noise_detected = False
        self.noise_detected_frame = None
        self.prev_speed = 0.0

        self.udp_ip = udp_ip
        self.udp_port = udp_port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.last_sent_speed = None
        self.speed_change_threshold = 0.01
        self.last_update_time = 0.0
        self.frame_count = 0
        
        self.left_calib_frequency = 1.2
        self.right_calib_frequency = 1.3
        self.left_calib_height_movement = 0.10
        self.right_calib_height_movement = 0.12
        self.base_noise_score = 0.38
        self.prev_left_heel_height = None
        self.prev_right_heel_height = None
        self.last_crossing_time_left = None
        self.last_crossing_time_right = None
        
        self.weber_fraction = 0.05
        self.last_applied_speed = 0.0
        
        self.left_heel_buffer = deque(maxlen=30)
        self.right_heel_buffer = deque(maxlen=30)
        
        self.last_motion_time = time.time()
        self.motion_timeout = 2.0

    def set_calibration_results(self, left_crossing_threshold, right_crossing_threshold, base_noise_score, 
                               left_calib_frequency, right_calib_frequency, left_calib_height, right_calib_height):
        self.left_crossing_threshold = left_crossing_threshold
        self.right_crossing_threshold = right_crossing_threshold
        self.base_noise_score = base_noise_score
        self.left_calib_frequency = left_calib_frequency
        self.right_calib_frequency = right_calib_frequency
        self.left_calib_height_movement = left_calib_height
        self.right_calib_height_movement = right_calib_height
        print(f"Calibration Applied - "
              f"Base Noise Score: {self.base_noise_score:.2f}, "
              f"Left Calib Frequency: {self.left_calib_frequency:.2f}, Right Calib Frequency: {self.right_calib_frequency:.2f}, "
              f"Left Calib Height: {self.left_calib_height_movement:.2f}, Right Calib Height: {self.right_calib_height_movement:.2f}")

    def calculate_speed(self, height_movement, stride_freq, calib_height, calib_freq):
        """비선형 속도 계산"""
        height_ratio = height_movement / calib_height if calib_height > 0 else 1.0
        freq_ratio = stride_freq / calib_freq if calib_freq > 0 else 1.0
        target_speed = 2.78  # 캘리브레이션 목표 속도
        adjusted_speed = target_speed * (height_ratio ** 0.7) * (freq_ratio)  # 제곱근으로 변동 완화
        return min(adjusted_speed, self.max_speed_walking)

    def detect_frontal_speed(self):
        if len(self.left_heel_heights) < 30 or len(self.right_heel_heights) < 30:
            return 0.0, 0.0

        if len(self.left_heel_buffer) >= 30 and len(self.right_heel_buffer) >= 30:
            left_change = max(self.left_heel_buffer) - min(self.left_heel_buffer)
            right_change = max(self.right_heel_buffer) - min(self.right_heel_buffer)
            if left_change <= 0.02 and right_change <= 0.02:
                self.crossings_left.clear()
                self.right_crossings.clear()
                return 0.0, 0.0

        if len(self.crossings_left) >= 2 and len(self.right_crossings) >= 2:
            last_cross_left = list(self.crossings_left)[-2:]
            last_cross_right = list(self.right_crossings)[-2:]
            
            start_frame_left = last_cross_left[0][1]
            end_frame_left = last_cross_left[1][1]
            start_frame_right = last_cross_right[0][1]
            end_frame_right = last_cross_right[1][1]
            
            left_heights_array = list(self.left_heel_heights)
            right_heights_array = list(self.right_heel_heights)
            current_frame = self.frame_count
            
            left_start_idx = max(0, len(left_heights_array) - (current_frame - start_frame_left + 1))
            left_end_idx = max(0, len(left_heights_array) - (current_frame - end_frame_left + 1))
            right_start_idx = max(0, len(right_heights_array) - (current_frame - start_frame_right + 1))
            right_end_idx = max(0, len(right_heights_array) - (current_frame - end_frame_right + 1))
            
            if left_start_idx < left_end_idx < len(left_heights_array):
                left_between = left_heights_array[left_start_idx:left_end_idx + 1]
                height_movement_left = max(left_between) - min(left_between)
            else:
                height_movement_left = self.left_calib_height_movement  # 기본값으로 캘리브레이션 값 사용
            if right_start_idx < right_end_idx < len(right_heights_array):
                right_between = right_heights_array[right_start_idx:right_end_idx + 1]
                height_movement_right = max(right_between) - min(right_between)
            else:
                height_movement_right = self.right_calib_height_movement  # 기본값으로 캘리브레이션 값 사용
            
            # 최근 5개 교차 간격 평균으로 Stride Frequency 계산
            left_intervals = [t2 - t1 for (t1, _), (t2, _) in zip(list(self.crossings_left)[:-1], list(self.crossings_left)[1:])][-5:]
            right_intervals = [t2 - t1 for (t1, _), (t2, _) in zip(list(self.right_crossings)[:-1], list(self.right_crossings)[1:])][-5:]
            avg_cross_left = np.mean(left_intervals) if left_intervals else 1.0
            avg_cross_right = np.mean(right_intervals) if right_intervals else 1.0
            
            left_stride_frequency = min(1.0 / avg_cross_left if avg_cross_left > 0 else 0, 4.5)  # 상한 4 Hz
            right_stride_frequency = min(1.0 / avg_cross_right if avg_cross_right > 0 else 0, 4.5)  # 상한 4 Hz
        
            # 속도 계산
            speed_left = self.calculate_speed(height_movement_left, left_stride_frequency, 
                                            self.left_calib_height_movement, self.left_calib_frequency)
            speed_right = self.calculate_speed(height_movement_right, right_stride_frequency, 
                                             self.right_calib_height_movement, self.right_calib_frequency)
            target_speed = (speed_left + speed_right) / 2

            if self.frame_count % 10 == 0:
                print(f"Debug: Left Speed: {speed_left:.2f}, Right Speed: {speed_right:.2f}, "
                      f"Stride Frequency Left: {left_stride_frequency:.2f}, Right: {right_stride_frequency:.2f}, "
                      f"Height Movement Left: {height_movement_left:.4f} (Frames: {start_frame_left}-{end_frame_left}), "
                      f"Height Movement Right: {height_movement_right:.4f} (Frames: {start_frame_right}-{end_frame_right})")

            return target_speed, max(left_stride_frequency, right_stride_frequency)
        return 0.0, 0.0

    def calculate_vibration(self):
        return 0.0

    def is_running(self):
        return False

    def detect_sit_stand_noise(self):
        if len(self.left_heel_40frame) < 40 or len(self.right_heel_40frame) < 40:
            return False

        recent_left_10frame = list(self.left_heel_40frame)[-10:]
        recent_right_10frame = list(self.right_heel_40frame)[-10:]
        
        left_x = np.array([pos[0] for pos in recent_left_10frame])
        left_y = np.array([pos[1] for pos in recent_left_10frame])
        left_z = np.array([pos[2] for pos in recent_left_10frame])
        right_x = np.array([pos[0] for pos in recent_right_10frame])
        right_y = np.array([pos[1] for pos in recent_right_10frame])
        right_z = np.array([pos[2] for pos in recent_right_10frame])

        delta_x_left = np.max(left_x) - np.min(left_x)
        delta_y_left = np.max(left_y) - np.min(left_y)
        delta_z_left = np.max(left_z) - np.min(left_z)
        delta_x_right = np.max(right_x) - np.min(right_x)
        delta_y_right = np.max(right_y) - np.min(right_y)
        delta_z_right = np.max(right_z) - np.min(right_z)

        score_left = 1 * delta_x_left + 1 * delta_y_left + 1 * delta_z_left
        score_right = 1 * delta_x_right + 1 * delta_y_right + 1 * delta_z_right
        noise_threshold = self.base_noise_score * 3.3

        recent_delta_y_left = delta_y_left
        recent_delta_y_right = delta_y_right
        movement_threshold = 0.02

        is_noise_candidate = (score_left > noise_threshold and score_right > noise_threshold and 
                            (recent_delta_y_left > movement_threshold or recent_delta_y_right > movement_threshold))
        
        if is_noise_candidate:
            self.noise_counter += 1
            if self.noise_counter >= 10:
                self.noise_detected = True
                self.noise_detected_frame = self.noise_detected_frame or self.frame_count
                print(f"Debug: Left Score: {score_left:.2f}, Right Score: {score_right:.2f}, Threshold: {noise_threshold:.2f}, "
                      f"Recent Delta Y Left: {recent_delta_y_left:.2f}, Right: {recent_delta_y_right:.2f}")
                return True
            return False
        else:
            self.noise_counter = max(0, self.noise_counter - 1)
            if self.noise_counter == 0 and self.noise_detected:
                left_heights_array = np.array(list(self.left_heel_heights))
                right_heights_array = np.array(list(self.right_heel_heights))
                dynamic_left_threshold = np.mean(left_heights_array) + np.std(left_heights_array) if len(left_heights_array) > 0 else 0.0
                dynamic_right_threshold = np.mean(right_heights_array) + np.std(right_heights_array) if len(right_heights_array) > 0 else 0.0
                if (abs(self.prev_left_heel_height - dynamic_left_threshold) < 0.1 and 
                    abs(self.prev_right_heel_height - dynamic_right_threshold) < 0.1 and 
                    0.02 <= recent_delta_y_left <= 0.10 and 0.02 <= recent_delta_y_right <= 0.10):
                    self.noise_detected = False
                    self.noise_detected_frame = None
                    print(f"Frame {self.frame_count}: Noise cleared, resuming speed calculation")
            return False

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

        # 동적 임계값 계산
        left_heights_array = np.array(list(self.left_heel_heights))
        right_heights_array = np.array(list(self.right_heel_heights))
        dynamic_left_threshold = np.mean(left_heights_array) + np.std(left_heights_array) if len(left_heights_array) > 0 else 0.0
        dynamic_right_threshold = np.mean(right_heights_array) + np.std(right_heights_array) if len(right_heights_array) > 0 else 0.0

        # 교차 감지 및 프레임 번호 저장 (10프레임 간격 조건 강화)
        if self.prev_left_heel_height is not None:
            left_cross = (self.prev_left_heel_height < dynamic_left_threshold <= left_heel_height)
            if left_cross:
                if self.last_crossing_time_left is not None:
                    interval = current_time - self.last_crossing_time_left
                    last_frame_left = self.crossings_left[-1][1] if self.crossings_left else 0
                    frame_interval = self.frame_count - last_frame_left
                    if frame_interval >= 10:  # 최소 10프레임 간격
                        self.crossings_left.append((current_time, self.frame_count))
                        self.last_crossing_time_left = current_time
                        print(f"Frame {self.frame_count}: Left Crossing Detected (Upward), Interval: {interval:.3f}s, Threshold: {dynamic_left_threshold:.2f}")
                    else:
                        print(f"Frame {self.frame_count}: Left Crossing Ignored (Interval: {frame_interval} < 10 frames)")
                else:
                    self.last_crossing_time_left = current_time
                    self.crossings_left.append((current_time, self.frame_count))
                    print(f"Frame {self.frame_count}: Left Crossing Detected (Upward, First), Threshold: {dynamic_left_threshold:.2f}")

        if self.prev_right_heel_height is not None:
            right_cross = (self.prev_right_heel_height < dynamic_right_threshold <= right_heel_height)
            if right_cross:
                if self.last_crossing_time_right is not None:
                    interval = current_time - self.last_crossing_time_right
                    last_frame_right = self.right_crossings[-1][1] if self.right_crossings else 0
                    frame_interval = self.frame_count - last_frame_right
                    if frame_interval >= 10:  # 최소 10프레임 간격
                        self.right_crossings.append((current_time, self.frame_count))
                        self.last_crossing_time_right = current_time
                        print(f"Frame {self.frame_count}: Right Crossing Detected (Upward), Interval: {interval:.3f}s, Threshold: {dynamic_right_threshold:.2f}")
                    else:
                        print(f"Frame {self.frame_count}: Right Crossing Ignored (Interval: {frame_interval} < 10 frames)")
                else:
                    self.last_crossing_time_right = current_time
                    self.right_crossings.append((current_time, self.frame_count))
                    print(f"Frame {self.frame_count}: Right Crossing Detected (Upward, First), Threshold: {dynamic_right_threshold:.2f}")

        self.prev_left_heel_height = left_heel_height
        self.prev_right_heel_height = right_heel_height

        if self.frame_count % 10 == 0:
            print(f"Frame {self.frame_count}: Left Heel: {left_heel_height:.2f}, Right Heel: {right_heel_height:.2f}")

        if self.frame_count < 90:
            return 0.0, 0, 0

        if len(self.crossings_left) < 2 or len(self.right_crossings) < 2:
            return 0.0, 0, 0

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
            self.prev_speed = 0.0
            self.noise_detected = False
            self.noise_counter = 0
            self.send_speed_to_unity(warning=warning)
            return self.current_speed, 0

        is_noise = self.detect_sit_stand_noise()
        if is_noise and self.noise_detected:
            self.prev_speed = self.current_speed * 0.9
            self.current_speed = self.prev_speed
            self.last_applied_speed = 0.0
            self.send_speed_to_unity(warning=warning)
            print(f"Frame {self.frame_count}: Sit/Stand noise detected, Speed forced to {self.current_speed:.2f}")
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
            self.current_speed = 0.5 * self.current_speed + 0.5 * min(adjusted_speed, max_speed) if target_speed > 0 else 0.0
            self.last_applied_speed = target_speed
            self.prev_speed = self.current_speed

        if not (left_heel_visible or right_heel_visible):
            self.current_speed = self.target_speed = 0.0
            self.last_applied_speed = 0.0
            self.prev_speed = 0.0

        self.current_speed = max(0.0, self.current_speed * speed_multiplier)
        self.send_speed_to_unity(warning=warning)

        if self.frame_count % 10 == 0:
            print(f"Frame {self.frame_count}: Speed: {self.current_speed:.2f}, Target Speed: {target_speed:.2f}, "
                  f"Stride Frequency: {stride_frequency:.1f}")

        return self.current_speed, stride_frequency
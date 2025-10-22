#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import websockets
import json
import time
import threading
from typing import Optional
import cv2
import mediapipe as mp
import numpy as np
import math
import socket
import sys

WS_SUBPROTOCOL = 'v1.phonescoring.jd.ubisoft.com'
ACCEL_ACQUISITION_FREQ_HZ = 200.0
ACCEL_ACQUISITION_LATENCY = 0.0
ACCEL_MAX_RANGE = 8.0
FRAME_DURATION = 0.015

TARGET_WIDTH = 640
TARGET_HEIGHT = 480
FPS_SMOOTH = 0.12
REF_DIST_PIXELS = 220.0 
ACCEL_SCALE = 15.0
Y_AXIS_SCALE = 1.8 
WIIMOTE_LENGTH = 40 
WIIMOTE_WIDTH = 10 

COLOR_ORANGE = (0, 165, 255)

mp_pose = mp.solutions.pose

class VirtualController:

    def __init__(self, console_ip: str):
        self.pairing_url = f"ws://{console_ip}:8080/smartphone"
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.disconnected = False
        
        self.should_start_accelerometer = False
        self.number_of_accels_sent = 0
        
        self.accel_lock = threading.Lock()
        self.accel_data_buffer = []

    async def send_message(self, __class: str, data: dict = {}):
        if self.ws and not self.disconnected:
            msg = {'root': {'__class': __class}}
            if data:
                msg['root'].update(data)
            
            try:
                await self.ws.send(json.dumps(msg, separators=(',', ':')))
            except Exception:
                self.disconnected = True

    async def on_message(self, raw_message: str):
        try:
            message = json.loads(raw_message)
            if '__class' not in message:
                return

            __class = message.get('__class')

            if __class == 'JD_PhoneDataCmdHandshakeContinue':
                await self.send_message('JD_PhoneDataCmdSync', {'phoneID': message['phoneID']})
            
            elif __class == 'JD_PhoneDataCmdSyncEnd':
                await self.send_message('JD_PhoneDataCmdSyncEnd', {'phoneID': message['phoneID']})

            elif __class == 'JD_EnableAccelValuesSending_ConsoleCommandData':
                self.should_start_accelerometer = True
                self.number_of_accels_sent = 0
            
            elif __class == 'JD_DisableAccelValuesSending_ConsoleCommandData':
                self.should_start_accelerometer = False
                with self.accel_lock:
                    self.accel_data_buffer.clear()
            
        except Exception:
            pass

    async def send_hello(self):
        await self.send_message('JD_PhoneDataCmdHandshakeHello', {
            'accelAcquisitionFreqHz': ACCEL_ACQUISITION_FREQ_HZ,
            'accelAcquisitionLatency': ACCEL_ACQUISITION_LATENCY,
            'accelMaxRange': ACCEL_MAX_RANGE,
        })

        try:
            async for message in self.ws:
                await self.on_message(message)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.disconnected = True

    async def tick_accelerometer(self):
        while not self.disconnected:
            if self.should_start_accelerometer:
                accel_data_to_send = []
                
                with self.accel_lock:
                    if self.accel_data_buffer:
                        accel_data_to_send = self.accel_data_buffer.copy()
                        self.accel_data_buffer.clear()
                
                if accel_data_to_send:
                    await self.send_message('JD_PhoneScoringData', {
                        'accelData': accel_data_to_send,
                        'timeStamp': self.number_of_accels_sent,
                    })
                    self.number_of_accels_sent += len(accel_data_to_send)
                
                await asyncio.sleep(FRAME_DURATION * 3)
            else:
                await asyncio.sleep(0.1)
        
        if self.ws and not self.ws.closed:
            try:
                await self.ws.close()
            except Exception:
                pass

    async def connect(self):
        try:
            async with websockets.connect(
                self.pairing_url,
                subprotocols=[WS_SUBPROTOCOL],
                ping_timeout=None
            ) as websocket:
                self.ws = websocket
                
                await asyncio.gather(
                    self.send_hello(),
                    self.tick_accelerometer()
                )
        
        except (OSError, websockets.exceptions.InvalidURI, websockets.exceptions.InvalidHandshake) as e:
            raise ConnectionError(f"Falha ao conectar: {e}")
        except Exception:
            pass
        finally:
            self.disconnected = True

def webcam_accelerometer_thread(controller: VirtualController):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        controller.disconnected = True
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except:
        pass

    pose = mp_pose.Pose(model_complexity=0, enable_segmentation=False,
                        min_detection_confidence=0.5, min_tracking_confidence=0.5)

    last_pos_vec = None
    window = "Controlador Webcam Just Dance"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, TARGET_WIDTH, TARGET_HEIGHT)
    
    jd_x, jd_y, jd_z = 0.0, 0.0, 0.0
    fps = 0.0
    last_time = time.time()

    while not controller.disconnected:
        ret, frame = cap.read()
        now = time.time()
        if not ret:
            time.sleep(0.01)
            continue
        
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        has_hand = False

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            
            rw = lm[mp_pose.PoseLandmark.LEFT_WRIST]
            rs = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
            rf = lm[mp_pose.PoseLandmark.LEFT_INDEX]

            if all(p.visibility > 0.4 for p in [rw, rs, rf]):
                xw = rw.x * TARGET_WIDTH
                yw = rw.y * TARGET_HEIGHT
                xs = rs.x * TARGET_WIDTH
                ys = rs.y * TARGET_HEIGHT
                xf = rf.x * TARGET_WIDTH 
                yf = rf.y * TARGET_HEIGHT 

                cv2.line(frame, (int(xs), int(ys)), (int(xw), int(yw)), COLOR_ORANGE, 2, lineType=cv2.LINE_AA)
                
                hand_vec_x = xf - xw
                hand_vec_y = yf - yw
                angle = math.atan2(hand_vec_y, hand_vec_x)
                
                perp_angle = angle + math.pi / 2
                dx_width = math.cos(perp_angle) * WIIMOTE_WIDTH / 2
                dy_width = math.sin(perp_angle) * WIIMOTE_WIDTH / 2
                
                dx_len = math.cos(angle) * WIIMOTE_LENGTH
                dy_len = math.sin(angle) * WIIMOTE_LENGTH

                p1 = np.array([xw - dx_width, yw - dy_width], dtype=np.int32)
                p2 = np.array([xw + dx_width, yw + dy_width], dtype=np.int32)
                p3 = np.array([p2[0] + dx_len, p2[1] + dy_len], dtype=np.int32)
                p4 = np.array([p1[0] + dx_len, p1[1] + dy_len], dtype=np.int32)
                
                points = np.array([p1, p2, p3, p4], dtype=np.int32)
                cv2.polylines(frame, [points], isClosed=True, color=COLOR_ORANGE, thickness=2, lineType=cv2.LINE_AA)

                has_hand = True

                dx = (rf.x - 0.5) * 2.0
                dy = (rf.y - 0.5) * 2.0
                dist_px = math.hypot(xw - xs, yw - ys)
                dz = np.clip(dist_px / REF_DIST_PIXELS, 0.0, 1.0) 

                pos_vec = np.array([dx, dy, dz], dtype=np.float32)
                
                if last_pos_vec is not None:
                    motion_x = 0.0; motion_y = 0.0; motion_z = 0.0
                    base_x = 0.0; base_y = 0.0; base_z = -1.0 
                    
                    raw_accel = (pos_vec - last_pos_vec) * ACCEL_SCALE
                    accel = np.clip(raw_accel, -4.0, 4.0)
                    motion_x = accel[0]
                    motion_y = -accel[1] * Y_AXIS_SCALE 
                    motion_z = accel[2]
                    
                    base_x = pos_vec[0] * 1.5 
                    base_y = -pos_vec[1] * 1.5 
                    xy_mag_sq = (base_x**2 + base_y**2)
                    if xy_mag_sq >= 1.0: base_z = -0.1 
                    else: base_z = -math.sqrt(1.0 - xy_mag_sq) 
                    
                    final_x = np.clip(base_x + motion_x, -8.0, 8.0)
                    final_y = np.clip(base_y + motion_y, -8.0, 8.0)
                    final_z = np.clip(base_z + motion_z, -8.0, 8.0) 
                    
                    jd_x = final_z 
                    jd_y = final_x 
                    jd_z = final_y 

                    jd_accel_tuple = (jd_x, jd_y, jd_z)

                    if controller.should_start_accelerometer:
                        with controller.accel_lock:
                            controller.accel_data_buffer.append(jd_accel_tuple)
                            controller.accel_data_buffer.append(jd_accel_tuple)
                            controller.accel_data_buffer.append(jd_accel_tuple)
                
                last_pos_vec = pos_vec.copy()
            else:
                last_pos_vec = None 

        y_offset = 30
        
        delta_time = now - last_time
        if delta_time > 0:
            inst_fps = 1.0 / delta_time
            fps = (1 - FPS_SMOOTH) * fps + FPS_SMOOTH * inst_fps if fps > 0 else inst_fps
        last_time = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_ORANGE, 1, cv2.LINE_AA)
        y_offset += 25

        if not has_hand:
            last_pos_vec = None
            cv2.putText(frame, "Mao nao detectada", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_ORANGE, 1, cv2.LINE_AA)
            y_offset += 25
            jd_x, jd_y, jd_z = 0.0, 0.0, 0.0

        final_text = f"JD-> X:{jd_x:+.1f} Y:{jd_y:+.1f} Z:{jd_z:+.1f}"
        cv2.putText(frame, final_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_ORANGE, 1, cv2.LINE_AA)
        y_offset += 25
        cv2.putText(frame, "By Comera", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_ORANGE, 1, cv2.LINE_AA)
        cv2.imshow(window, frame)
        
        key_pressed = cv2.waitKey(1) & 0xFF
        window_closed = False
        try:
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                window_closed = True
        except cv2.error:
            window_closed = True

        if key_pressed == ord('q') or window_closed:
            controller.disconnected = True
            break

    pose.close()
    cap.release()
    cv2.destroyAllWindows()
    controller.disconnected = True 

def get_local_ip_prefix():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(1.0)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        prefix = ".".join(ip.split('.')[:-1]) + "."
        return prefix
    except Exception:
        return "192.168.1."
    finally:
        s.close()

async def find_console_ip(prefix: str):
    tasks = []
    for i in range(100, 151):
         ip = f"{prefix}{i}"
         tasks.append(check_ip(ip))
    for i in list(range(1, 100)) + list(range(151, 255)):
        ip = f"{prefix}{i}"
        tasks.append(check_ip(ip))
    
    results = await asyncio.gather(*tasks)
    found_ips = [ip for ip in results if ip]
    
    if found_ips:
        return found_ips[0]
    else:
        common_prefixes = ["192.168.0.", "192.168.15.", "10.0.0."]
        if prefix in common_prefixes: common_prefixes.remove(prefix)
        
        for next_prefix in common_prefixes:
            tasks = []
            for i in range(1, 255):
                ip = f"{next_prefix}{i}"
                tasks.append(check_ip(ip))
            results = await asyncio.gather(*tasks)
            found_ips = [ip for ip in results if ip]
            if found_ips:
                return found_ips[0]

    return None

async def check_ip(ip: str):
    try:
        url = f"ws://{ip}:8080/smartphone"
        await asyncio.wait_for(
            websockets.connect(url, subprotocols=[WS_SUBPROTOCOL], open_timeout=0.4, close_timeout=0.1),
            timeout=0.5
        )
        return ip
    except Exception:
        return None

async def async_main():
    console_ip = None
    
    if len(sys.argv) > 1 and sys.argv[1].startswith('--'):
        console_ip = sys.argv[1][2:]
    
    if not console_ip:
        ip_prefix = get_local_ip_prefix()
        console_ip = await find_console_ip(ip_prefix)
    
    if not console_ip:
        return

    controller = VirtualController(console_ip)

    t_webcam = threading.Thread(target=webcam_accelerometer_thread, args=(controller,), daemon=True)
    t_webcam.start()

    try:
        await controller.connect()
    except ConnectionError:
        controller.disconnected = True
    
    if t_webcam.is_alive():
        t_webcam.join(timeout=2.0)

if __name__ == "__main__":
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        pass
import cv2
import mediapipe as mp
import time
import numpy as np
import os
from hand_tracker import hand_tracker



class virtual_painter:
	def __init__(self):
		self.image_handler = cv2.VideoCapture(0)
		self.image_handler.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
		self.image_handler.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

		self.image_overlays = [cv2.imread(f'virtual_painter_designs/{image}') for image in os.listdir('virtual_painter_designs')]
		self.overlay_index = 0
		self.colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 255)]
		self.circle_positions = []
		self.is_draw = True

		self.hand_tracker = hand_tracker(detection_confidence = 0.6)


	def take_picture(self):
		status, frame = self.image_handler.read()
		frame = cv2.flip(frame, 1)

		if status:
			frame[0:102, 0:480] = self.image_overlays[self.overlay_index]
			
			return frame


	def fingers_up(self, landmarks):
		fingers_up = []

		for i in range(8, 24, 4):
			if landmarks[i][1].y < landmarks[i - 2][1].y:
				fingers_up.append(i)

		return fingers_up


	def process_image(self, frame):
		landmarks, results = self.hand_tracker.find_hand_positions(frame)
		frame = self.hand_tracker.draw_landmarks(frame, results)

		if len(landmarks) > 0:
			h, w, c = frame.shape
			fingers_up = self.fingers_up(landmarks)

			if fingers_up == [8]:
				index_x, index_y = int(landmarks[8][1].x * w), int(landmarks[8][1].y * h)
				cv2.circle(frame, (index_x, index_y), 5, self.colors[self.overlay_index], 30)

				if self.is_draw:
					if [index_x, index_y, self.colors[self.overlay_index]] not in self.circle_positions:
						self.circle_positions.append([index_x, index_y, self.colors[self.overlay_index]])

				else:
					for i in range(0, len(self.circle_positions)):
						try:
							if abs(self.circle_positions[i][0] - index_x) < 30 and abs(self.circle_positions[i][1] - index_y) < 30:
								self.circle_positions.pop(i)
						except:
							pass

			elif fingers_up == [8, 12]:
				index_x, index_y = int(landmarks[8][1].x * w), int(landmarks[8][1].y * h)
				middle_x, middle_y = int(landmarks[12][1].x * w), int(landmarks[12][1].y * h)

				if middle_y < 150:
					if middle_x < 120:
						self.overlay_index = 3
						self.is_draw = False
					
					elif middle_x < 240 and middle_x > 120:
						self.overlay_index = 2
						self.is_draw = True 

					elif middle_x < 360 and middle_x > 240:
						self.overlay_index = 1
						self.is_draw = True 

					elif middle_x < 480 and middle_x > 360:
						self.overlay_index = 0
						self.is_draw = True
	


		for x, y, color in self.circle_positions:
			cv2.circle(frame, (x, y), 5, color, 10)


		return frame





if __name__ == '__main__':
	VirtualPainter = virtual_painter()
	run = True

	while run:
		frame = VirtualPainter.take_picture()
		processed_frame = VirtualPainter.process_image(frame)
		cv2.imshow('Virtual Painter', processed_frame)


		if cv2.waitKey(1) & 0xFF == ord('q'):
			run = False
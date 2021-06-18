import cv2
import mediapipe as mp
import time



class hand_tracker:
	def __init__(self, mode = False, max_hands = 2, detection_confidence = 0.5, track_confidence = 0.5):
		self.mode = mode
		self.max_hands = max_hands
		self.detection_confidence = detection_confidence
		self.track_confidence = track_confidence
		self.drawer = mp.solutions.drawing_utils

		self.hand_tracker = mp.solutions.hands.Hands(self.mode, self.max_hands, self.detection_confidence, self.track_confidence)


	def find_hand_positions(self, image, return_results = True):
		rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		results = self.hand_tracker.process(rgb_image)
		landmarks, results_list = [], []

		if results.multi_hand_landmarks:
			for result in results.multi_hand_landmarks:
				results_list.append(result)
				for id, landmark in enumerate(result.landmark):
					landmarks.append([id, landmark])


		if return_results:
			return (landmarks, results_list)
		else:
			return landmarks


	def draw_landmarks(self, frame, results):
		if len(results) > 0:
			for result in results: 
				self.drawer.draw_landmarks(frame, result, mp.solutions.hands.HAND_CONNECTIONS)

		return frame




def main():
	handTracker = hand_tracker()
	image_handler = cv2.VideoCapture(0)
	drawer = mp.solutions.drawing_utils
	run = True

	while run:
		staus, image = image_handler.read()
		landmark_positions, results = handTracker.find_hand_positions(image)
		
		for result in results: drawer.draw_landmarks(image, result, mp.solutions.hands.HAND_CONNECTIONS)

		cv2.imshow('Image', image)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			run = False



if __name__ == '__main__':
	main()




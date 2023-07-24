import mediapipe as mp
import cv2
import numpy as np
import threading
import pygame
from pygame.locals import *
from gtts import gTTS
from playsound import playsound
import time

from workout import validate_pushup, validate_squat, validate_situp, calc_angle
# tts


def tts(msg):
    tts = gTTS(msg, lang='en', tld='com.au')
    tts.save('audio.mp3')
    playsound('audio.mp3', block=False)


def is_activation_keyword(command):
    activation_keywords = ['hey computer', 'hello computer']
    for keyword in activation_keywords:
        if keyword in command:
            return True
    return False


positions = {
    'nose': [],

    'right_shoulder': [],
    'left_shoulder': [],

    'right_elbow': [],
    'left_elbow': [],

    'right_wrist': [],
    'left_wrist': [],

    'right_hip': [],
    'left_hip': [],

    'right_knee': [],
    'left_knee': [],

    'right_ankle': [],
    'left_ankle': [],

    'right_foot_index': [],

    'visibility': {
        'right_knee': 0,
        'right_ankle': 0,
        'right_hip': 0
    }
}

thresholds = {
    "pushup": {
        "down": 80,
        "up": 140,
        "flat_body": 140,
        "horizontal_to_ground": range(-30, 30)
    },
    "squat": {
        "down": 90,
        "up": 160,
        "knees_over": -20
    },
    "situp": {
        "down": 160,
        "up": 50
    }
}

# down, up, count
pushup_completion = [False, False, 0]
squat_completion = [False, False, 0]
situp_completion = [False, False, 0]

# MODEL
model_path = 'pose_landmarker_heavy.task'
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

pose_landmarker = mp_pose.Pose(
    static_image_mode=False,
    enable_segmentation=True,
    smooth_landmarks=True,
    min_detection_confidence=0.9,
    min_tracking_confidence=0.9,
)


def process_frames(frame):

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose_landmarker.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # mp_draw.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # returning annotated image and landmark list
    return image, results.pose_landmarks


def get_positions(landmark_list, changex):
    landmark_mapping = {
        mp_pose.PoseLandmark.NOSE: 'nose',
        mp_pose.PoseLandmark.RIGHT_SHOULDER: 'right_shoulder',
        mp_pose.PoseLandmark.LEFT_SHOULDER: 'left_shoulder',
        mp_pose.PoseLandmark.RIGHT_ELBOW: 'right_elbow',
        mp_pose.PoseLandmark.LEFT_ELBOW: 'left_elbow',
        mp_pose.PoseLandmark.RIGHT_WRIST: 'right_wrist',
        mp_pose.PoseLandmark.LEFT_WRIST: 'left_wrist',
        mp_pose.PoseLandmark.RIGHT_HIP: 'right_hip',
        mp_pose.PoseLandmark.LEFT_HIP: 'left_hip',
        mp_pose.PoseLandmark.RIGHT_KNEE: 'right_knee',
        mp_pose.PoseLandmark.LEFT_KNEE: 'left_knee',
        mp_pose.PoseLandmark.RIGHT_ANKLE: 'right_ankle',
        mp_pose.PoseLandmark.LEFT_ANKLE: 'left_ankle',
        mp_pose.PoseLandmark.RIGHT_FOOT_INDEX: 'right_foot_index'
    }

    for landmark, key in landmark_mapping.items():
        positions[key] = [landmark_list.landmark[landmark].x *
                          width+changex, landmark_list.landmark[landmark].y * height]
        
    visibility_mapping = {
        'right_knee': landmark_list.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].visibility,
        'right_ankle': landmark_list.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].visibility,
        'right_hip': landmark_list.landmark[mp_pose.PoseLandmark.RIGHT_HIP].visibility
    }

    for key, value in visibility_mapping.items():
        positions['visibility'][key] = value


def workout_pushup():
    form_status = None
    down = validate_pushup(positions, thresholds, "down")
    up = validate_pushup(positions, thresholds, "up")

    #down -> valid
    if down == True and not pushup_completion[0]:
        pushup_completion[0] = True
        tts_thread = threading.Thread(target=tts, args=("Down",))
        tts_thread.start()
    else:
        form_status = down
        
    #up -> valid : pushup completed
    if pushup_completion[0] and up == True:
        pushup_completion[1] = True
        tts_thread = threading.Thread(target=tts, args=("Up",))
        tts_thread.start()
        
        pushup_completion[2] += 1
        return True
    
    else:
        form_status = up
        
    return form_status
    



def workout_squat():
    form_status = None
    down = validate_squat(positions, thresholds, "down")
    up = validate_squat(positions, thresholds, "up")
    
    if down == True and not squat_completion[0]:
        squat_completion[0] = True
        tts_thread = threading.Thread(target=tts, args=("Down",))
        tts_thread.start()
    else:
        form_status = down
        
    if up == True and squat_completion[0]:
        squat_completion[1] = True
        tts_thread = threading.Thread(target=tts, args=("Up",))
        tts_thread.start()
        
        squat_completion[2] += 1
        return True
    else:
        form_status = up
        
    return form_status


def workout_situp():
    form_status = None
    down = validate_situp(positions, thresholds, "down")
    up = validate_situp(positions, thresholds, "up")
    
    if down == True and not situp_completion[0]:
        situp_completion[0] = True
        tts_thread = threading.Thread(target=tts, args=("Down",))
        tts_thread.start()
    else:
        form_status = down
        
    if up == True and situp_completion[0]:
        situp_completion[1] = True
        tts_thread = threading.Thread(target=tts, args=("Up",))
        tts_thread.start()
        
        situp_completion[2] += 1
        return True
    else:
        form_status = up
        
    return form_status

def draw_character(screen, positions):

    circle_positions = ['nose', 'right_shoulder', 'left_shoulder', 'right_elbow', 'left_elbow', 'right_wrist',
                        'left_wrist', 'right_hip', 'left_hip', 'right_knee', 'left_knee', 'right_ankle', 'left_ankle']
    circle_color = (255, 255, 255)
    circle_radius = 5
    connections = [
        ('nose', 'right_shoulder'),
        ('right_shoulder', 'right_elbow'),
        ('right_elbow', 'right_wrist'),
        ('right_shoulder', 'right_hip'),
        ('right_hip', 'right_knee'),
        ('right_knee', 'right_ankle'),
        ('nose', 'left_shoulder'),
        ('left_shoulder', 'left_elbow'),
        ('left_elbow', 'left_wrist'),
        ('left_shoulder', 'left_hip'),
        ('left_hip', 'left_knee'),
        ('left_knee', 'left_ankle')
    ]

    line_color = (255, 255, 255)
    line_thickness = 2

    # Draw circles
    for position in circle_positions:
        pygame.draw.circle(
            screen, circle_color, (positions[position][0], positions[position][1]), circle_radius)

    # Draw lines
    for connection in connections:
        start_pos = (positions[connection[0]][0], positions[connection[0]][1])
        end_pos = (positions[connection[1]][0], positions[connection[1]][1])
        pygame.draw.line(screen, line_color, start_pos,
                         end_pos, line_thickness)


def render_text(surface, text, position):
    font = pygame.font.Font(None, 24)
    text_surface = font.render(text, True, (255, 255, 255))
    surface.blit(text_surface, position)


# VIDEO
# width,height = 720,1280q
cap = cv2.VideoCapture(0)
pygame.init()
pygame.display.set_caption("OpenCV camera stream on Pygame")
screen = pygame.display.set_mode([1280, 720])
clock = pygame.time.Clock()

changex = 0
changex_direction = 0


# Define movement speed and duration
movement_speed = 80
key_press_start_time = None
movement_duration = .1

while cap.isOpened():
    ret, frame = cap.read()
    width, height = frame.shape[1], frame.shape[0]
    annotated_image = np.copy(frame)

    # process frame and display landmarks
    annotated_image, landmark_list = process_frames(annotated_image)

    # display windows
    # cv2.imshow('Mediapipe Feed', annotated_image)
    screen.fill([0, 0, 0])

    if landmark_list:
        thread = threading.Thread(target=get_positions, args=(landmark_list, changex)).start()
        
        draw_character(screen, positions)

        # pushup_status = workout_pushup() 
        # if pushup_status == True: 
        #     pushup_completion[0] , pushup_completion[1] = False, False
        # else:
        #     form_status = pushup_status


        squat_status = workout_squat()
        if squat_status == True: 
            squat_completion[0], squat_completion[1] = False, False
        else: 
            form_status = squat_status


        # situp_status = workout_situp()
        # if situp_status == True: 
        #     situp_completion[0], situp_completion[1] = False, False
        # else:
        #     form_status = situp_status

    else:
        form_status = "No Person Detected"
     
    render_text(screen, 'Status: ' +
                    str(form_status), (25, 60))  
    render_text(screen, 'Pushups Completed: ' +
                str(pushup_completion[2]), (25, 100))
    render_text(screen, 'Squats Completed: ' +
                str(squat_completion[2]), (25, 140))
    render_text(screen, 'Situps Completed: ' +
                str(situp_completion[2]), (25, 180))

    if cv2.waitKey(1) == ord('q'):
        break

    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_a:
                key_press_start_time = time.time()
                changex_direction = -1

            if event.key == pygame.K_d:
                key_press_start_time = time.time()
                changex_direction = 1

        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_a or event.key == pygame.K_d:
                changex_direction = 0

    if key_press_start_time is not None:
        elapsed_time = time.time() - key_press_start_time
        if elapsed_time >= movement_duration:
            key_press_start_time = None
            changex += movement_speed * changex_direction * movement_duration
        else:
            changex += movement_speed * changex_direction * elapsed_time

    keys = pygame.key.get_pressed()
    if keys[pygame.K_a]:
        changex_direction = -1
    elif keys[pygame.K_d]:
        changex_direction = 1
    else:
        changex_direction = 0

    changex += movement_speed * changex_direction



    clock.tick(60)
    pygame.display.update()

cap.release()
cv2.destroyAllWindows()

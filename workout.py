import math
def calc_angle(B,A,C):
    """
        B
    A
        C
    """
    
    
    
    AB = math.sqrt(math.pow(A[0]-B[0],2) + math.pow(A[1]-B[1],2))
    BC = math.sqrt(math.pow(B[0]-C[0],2) + math.pow(B[1]-C[1],2))
    AC = math.sqrt(math.pow(A[0]-C[0],2) + math.pow(A[1]-C[1],2))
    
    #law of cosines
    BAC = math.acos((math.pow(BC, 2) - math.pow(AB, 2) - math.pow(AC, 2)) / (-2 * AB * AC))
    BAC = math.degrees(BAC)
    return BAC
    
def validate_pushup(positions, thresholds, state):
    try:
        
        #valid pushup angles
        angle_ARMS = calc_angle(positions['right_shoulder'], positions['right_elbow'], positions['right_wrist'])
        angle_HIP = calc_angle(positions['right_shoulder'], positions['right_hip'], positions['right_knee'])
        angle_HIP2 = calc_angle(positions['right_shoulder'], positions['right_hip'], positions['right_ankle'])
        
        #horizontal to ground
        horizontal = positions['right_foot_index'][1]-positions['right_wrist'][1]
        if positions['visibility']['right_knee'] >= .5 and positions['visibility']['right_ankle'] >= .5:
            if int(horizontal) in thresholds['pushup']['horizontal_to_ground']:
                if angle_HIP >= thresholds['pushup']['flat_body'] and angle_HIP2 >= thresholds['pushup']['flat_body']:
                    if state == "down":
                        if angle_ARMS <= thresholds['pushup'][state]:
                            return True
                    if state == "up":
                        if angle_ARMS >= thresholds['pushup'][state]:
                            return True
                    else:
                        return "Keep your body straight (180˚)"
                else:
                    return "Get horizontal to the ground"
        else:
            return "Body not in frame"
            
    except Exception as e:
        print("Coords not found", e)
        
def validate_squat(positions, thresholds, state):
    angle_KNEES = calc_angle(positions['right_hip'], positions['right_knee'], positions['right_ankle'])
    if positions['visibility']['right_knee'] >= .5 and positions['visibility']['right_ankle'] >= .5 and positions['visibility']['right_hip'] >= .5:
        if state == "down":
            knees_over = (positions['right_knee'][0] - positions['right_ankle']) <= thresholds['squat']['knees_over']
            if angle_KNEES <= thresholds['squat'][state] and knees_over:
                return True
        if state == "up":
            if angle_KNEES >= thresholds['squat'][state]:
                return True
    else:
        return "Body not in frame"
    
    
def validate_situp(positions,thresholds, state):
    angle_HIP = calc_angle(positions['right_shoulder'], positions['right_hip'], positions['right_knee'])
    if positions['visibility']['right_knee'] >= .5 and positions['visibility']['right_hip'] >= .5:
        if state == "down":
            if angle_HIP <= thresholds['situp'][state]:
                return True
        if state == "up":
            if angle_HIP >= thresholds['situp'][state]:
                return True
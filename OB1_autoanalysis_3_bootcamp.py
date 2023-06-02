'''
---
# **OB1 AutoAnalysis 1-3) Joint Angle ROM**
---

### **_Use for all ANALYSIS since cleaned data is all SAME FORMAT_**

Soowan Choi
'''


from OB1_autoanalysis_3_functions import * # todo import other modules


def load_op(op_file):
  # create dataframe from uploaded csv files using pandas.read_csv()
  op = pd.read_csv(op_file) 
  return op 


def load_ma(ma_file):
  # create dataframe from uploaded csv files using pandas.read_csv() & skip the first few rows (3) of information
  ma = pd.read_csv(ma_file) 
  return ma


# Select Game
# op_games = ['Sqt', 'StLun', 'VMODip', 'HipFlex', 'HipExt', 'HipAbd', 'Kick', 'LatStep', 'BackStep',
#             'StarJump', 'Run',
#             'SeatKnExt', 'SeatHipFlex', 'SeatStarJump',
#             'SeatClfStr', 'ForStep', 'CalfStr', 'TdemStnce']
# ma_games = ['Sqt', 'StLun', 'VMODip', 'HipFlex', 'HipExt', 'HipAbd', 'Kick', 'LatStep', 'BackStep',
#             'StarJump', 'Run',
#             'SeatKnExt', 'SeatHipFlex', 'SeatStarJump',
#             'SeatClfStr', 'ForStep', 'CalfStr', 'TdemStnce']
op_games = ['Sqt', 'StLun', 'VMODip', 'HipFlex', 'HipExt', 'HipAbd', 'Kick', 'LatStep', 'BackStep',
            'StarJump', 'Run',
            'SeatKnExt', 'SeatHipFlex', 'SeatStarJump',
            'SeatClfStr', 'ForStep', 'CalfStr', 'TdemStnce']
ma_games = ['Sqt', 'StLun', 'VMODip', 'HipFlex', 'HipExt', 'HipAbd', 'Kick', 'LatStep', 'BackStep',
            'StarJump', 'Run',
            'SeatKnExt', 'SeatHipFlex', 'SeatStarJump',
            'SeatClfStr', 'ForStep', 'CalfStr', 'TdemStnce']


# SELECT FILES HERE
mmdd = '0402' 
p = 'P07'
mmdd_p = mmdd + '_' + p

directory_unknown = []

for game_ind in range(len(op_games)):
    print(f'\n\n\n\n\n\n\n\n{op_games[game_ind]}\n\n\n\n\n\n\n\n')
    # *** For each participant rename BC#-Game ***
    # Load bootcamp.csv file to rename
    bootcamp = pd.read_csv("/Users/soowan/Documents/VSCODE/Pearl/bootcamp.csv") 

    # For each row 
    # If correct participant
    # For each column
    # If correct Game and cell isn't empty
    # Rename: BC#-Game

    temp = []
    for i in range(len(bootcamp)):
        if mmdd_p in bootcamp.iloc[i,0]:
            for j in range(len(bootcamp.columns)):
                if (op_games[game_ind] in str(bootcamp.iloc[i,j])) and (str(bootcamp.iloc[i,j]) != 'nan'):
                    op_game = bootcamp.columns[j] + '-' + op_games[game_ind]
                    ma_game = bootcamp.columns[j] + '-' + ma_games[game_ind]
                    break
                else:
                    op_game = 'NA' + '-' + op_games[game_ind]
                    ma_game = 'NA' + '-' + op_games[game_ind]
                

    op_file = '2023' + mmdd_p[:4] + '-' + mmdd_p[-3:] + '-' + op_game + "-Data-OP-CLEAN.csv"
    ma_file = '2023' + mmdd_p[:4] + '-' + mmdd_p[-3:] + '-' + ma_game + "-MA-CLEAN.csv"

    try:
        # Load OP Data
        op = load_op('/Users/soowan/Documents/PEARL/Data/Data_0551/2023_' + mmdd_p + '/Clean_' + mmdd_p + '/' + op_file)
        print(op.head(3))

        # Load MA Data
        ma = load_ma('/Users/soowan/Documents/PEARL/Data/Data_0551/2023_' + mmdd_p + '/Clean_' + mmdd_p + '/' + ma_file)
        print(ma.head(3))

    except FileNotFoundError:
        # if directory game file doesn't exist, go to next game
        directory_unknown.append(op_file)
        continue


    op_filte = op.copy()
    ma_filte = ma.copy()
    op_cut, ma_cut = cut_data(op_filte, ma_filte)
    # align data using: METHOD 2
    op_align_joints, ma_align_joints = align_joints(op_cut, ma_cut)


    # Visualize ALL Data (39 graphs total)
    op_head = ['Head']
    ma_head = ['Front.Head']
    op_side = ['Left','Right']
    ma_side = ['L.','R.']
    xyz = ['Y','Z','X']
    # Head Data
    for i in range(len(op_head)):
        for k in range(len(xyz)):
            op_joint = op_head[i] + xyz[k]
            ma_joint = ma_head[i] + xyz[k]
            joint = ma_joint
            if xyz[k] == 'Y':
                data_vis(op_align_joints, ma_align_joints, joint, op_joint, ma_joint)  # align horizontal(Y) coordinate
            elif xyz[k] == 'Z':
                data_vis(op_align_joints, ma_align_joints, joint, op_joint, ma_joint)  # align vertical(Z) coordinate
            elif xyz[k] == 'X':
                data_vis(op_align_joints, ma_align_joints, joint, op_joint, ma_joint)  # align depth(X) coordinate


    op_final = op_align_joints.copy().reset_index().drop("index",axis=1)
    ma_final = ma_align_joints.copy().reset_index().drop("index",axis=1)


    # 1-3) Min/Max Joint Angle 


    # left angle
    op_elbow_ang_left, op_shoulder_ang_left, op_hip_ang_left, op_knee_ang_left = op_joint_angle(op_final, 'Left')
    ma_elbow_ang_left, ma_shoulder_ang_left, ma_hip_ang_left, ma_knee_ang_left = ma_joint_angle(ma_final, 'L')

    # right angle 
    op_elbow_ang_right, op_shoulder_ang_right, op_hip_ang_right, op_knee_ang_right = op_joint_angle(op_final, 'Right')
    ma_elbow_ang_right, ma_shoulder_ang_right, ma_hip_ang_right, ma_knee_ang_right = ma_joint_angle(ma_final, 'R')


    # left angle MIN/MAX + right angle MIN/MAX
    angle_table = pd.DataFrame()
    df_list = []

    op_left = [op_elbow_ang_left, op_shoulder_ang_left, op_hip_ang_left, op_knee_ang_left]
    ma_left = [ma_elbow_ang_left, ma_shoulder_ang_left, ma_hip_ang_left, ma_knee_ang_left]
    op_right = [op_elbow_ang_right, op_shoulder_ang_right, op_hip_ang_right, op_knee_ang_right]
    ma_right = [ma_elbow_ang_right, ma_shoulder_ang_right, ma_hip_ang_right, ma_knee_ang_right]

    leftright = ['L', 'R']
    jointname = ['Elbow', ' Shoulder', 'Hip', 'Knee']
    minmax = ['Min', 'Max']

    for i in leftright:
        for j in range(len(jointname)):
            for k in minmax:
                if i == 'L' and k == 'Min':
                    # compare the minimum angles
                    op_min, ma_min, diff, per = min_angle(op_left[j], ma_left[j])
                    single_vals = [op_min, ma_min, diff, per]
                elif i == 'L' and k == 'Max':
                    # compare the maximum angles
                    op_max, ma_max, diff, per = max_angle(op_left[j], ma_left[j])
                    single_vals = [op_max, ma_max, diff, per]
                elif i == 'R' and k == 'Min':
                    # compare the minimum angles
                    op_min, ma_min, diff, per = min_angle(op_right[j], ma_right[j])
                    single_vals = [op_min, ma_min, diff, per]
                elif i == 'R' and k == 'Max':
                    # compare the maximum angles
                    op_max, ma_max, diff, per = max_angle(op_right[j], ma_right[j])
                    single_vals = [op_max, ma_max, diff, per]
                    
                col = [f'OP({i}.{jointname[j]}.{k})', f'MA({i}.{jointname[j]}.{k})', 'Diff[deg]', 'Error[%]']
                tmp = pd.DataFrame([single_vals], columns = col)
                df_list.append(tmp)


    # show results
    joint_angles = table_angle_results(df_list)

    # DOWNLOAD the angle results --> paste into data results
    joint_angles.to_csv(rf'/Users/soowan/Downloads/2023{mmdd}-{p}-{op_game}-angle.csv', encoding = 'utf-8-sig') 

print("\nFOLLOWING FILES DO NOT EXIST:", directory_unknown)
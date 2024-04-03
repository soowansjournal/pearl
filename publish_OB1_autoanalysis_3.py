'''
---
# **OB1 AutoAnalysis 1-3) Joint Angle ROM**
---

### **_Use for all ANALYSIS since cleaned data is all SAME FORMAT_**

Soowan Choi
'''


from publish_OB1_autoanalysis_3_functions import * # todo import other modules


def load_op(op_file):
  # create dataframe from uploaded csv files using pandas.read_csv()
  op = pd.read_csv(op_file) 
  return op 


def load_ma(ma_file):
  # create dataframe from uploaded csv files using pandas.read_csv() & skip the first few rows (3) of information
  ma = pd.read_csv(ma_file) 
  return ma


# Select Game
# op_games = ['Wizards', 'War', 'Jet', 'Astro']
# ma_games = ['Wizards', 'War', 'Jet', 'Astro']

op_games = ['Wizards']
ma_games = ['Wizards']


# SELECT FILES HERE
mmdd = '0125' 
p = 'P??'
mmdd_p = mmdd + '_' + p

directory_unknown = []

for game_ind in range(len(op_games)):
    print(f'\n\n\n\n\n\n\n\n{op_games[game_ind]}\n\n\n\n\n\n\n\n')
    op_file = '2024' + mmdd + '-' + p + '-' + op_games[game_ind] + "-Data-OP-CLEAN.csv"
    ma_file = '2024' + mmdd + '-' + p + '-' + ma_games[game_ind] + "-MA-CLEAN.csv"
    #print(op_file, '\t', ma_file)

    try:
        # Load OP Data
        op = load_op('/Users/soowan/Downloads/' + op_file)
        print(op.head(3))

        # Load MA Data
        ma = load_ma('/Users/soowan/Downloads/' + ma_file)
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
    # op_joints = ['Shoulder','Elbow','Wrist','Hip','Knee','Foot','Hip','Knee','Foot']
    op_joints = ['Hip', 'Hip']
    # ma_joints = ['Shoulder','Elbow','Wrist','ASIS','Knee','Ankle','Hip_JC','Knee_JC','Ankle_JC']
    ma_joints = ['ASIS','Hip_JC']
    op_side = ['Left','Right']
    ma_side = ['L.','R.']
    xyz = ['Y','Z','X']

    # Head Data
    for i in range(len(op_head)):
        for k in range(len(xyz)):
            op_joint = op_head[i] + xyz[k]
            ma_joint = ma_head[i] + xyz[k]
            joint = ma_joint
            data_vis(op_align_joints, ma_align_joints, joint, op_joint, ma_joint)  # align depth(X) coordinatealign_joints, joint, op_joint, ma_joint)  # align depth(X) coordinate

    # Body Data
    for i in range(len(op_joints)):                   # for each joints
        for j in range(len(op_side)):                   # for each sides 
            for k in range(len(xyz)):                     # for each xyz 
                op_joint = op_joints[i] + op_side[j] + xyz[k]  # specific OP joint name
                joint = ma_side[j] + ma_joints[i] + xyz[k]     # joint of interest

                if "_JC" in ma_joints[i]:
                    ma_joint = "V_" + ma_side[j] + ma_joints[i] + xyz[k]  # specific MA joint name 
                else:
                    ma_joint = ma_side[j] + ma_joints[i] + xyz[k]  # specific MA joint name 
                
                data_vis(op_align_joints, ma_align_joints, ma_joint, op_joint, ma_joint)  # align coordinate

    op_final = op_align_joints.copy().reset_index().drop("index",axis=1)
    ma_final = ma_align_joints.copy().reset_index().drop("index",axis=1)


    # 1-3) Min/Max Joint Angle 


    # left angle
    op_elbow_ang_left, op_shoulder_ang_left, op_hip_ang_left, op_knee_ang_left = op_joint_angle(op_final, 'Left')
    ma_elbow_ang_left, ma_shoulder_ang_left, ma_hip_ang_left, ma_knee_ang_left, ma_shoulder_ang_left_JC, ma_hip_ang_left_JC, ma_knee_ang_left_JC = ma_joint_angle(ma_final, 'L')

    # right angle 
    op_elbow_ang_right, op_shoulder_ang_right, op_hip_ang_right, op_knee_ang_right = op_joint_angle(op_final, 'Right')
    ma_elbow_ang_right, ma_shoulder_ang_right, ma_hip_ang_right, ma_knee_ang_right, ma_shoulder_ang_right_JC, ma_hip_ang_right_JC, ma_knee_ang_right_JC = ma_joint_angle(ma_final, 'R')


    # left angle MIN/MAX + right angle MIN/MAX
    angle_table = pd.DataFrame()
    df_list = []

    op_left = [op_elbow_ang_left, op_shoulder_ang_left, op_hip_ang_left, op_knee_ang_left, op_shoulder_ang_left, op_hip_ang_left, op_knee_ang_left]
    ma_left = [ma_elbow_ang_left, ma_shoulder_ang_left, ma_hip_ang_left, ma_knee_ang_left, ma_shoulder_ang_left_JC, ma_hip_ang_left_JC, ma_knee_ang_left_JC]
    op_right = [op_elbow_ang_right, op_shoulder_ang_right, op_hip_ang_right, op_knee_ang_right, op_shoulder_ang_right, op_hip_ang_right, op_knee_ang_right]
    ma_right = [ma_elbow_ang_right, ma_shoulder_ang_right, ma_hip_ang_right, ma_knee_ang_right, ma_shoulder_ang_right_JC, ma_hip_ang_right_JC, ma_knee_ang_right_JC]

    leftright = ['L','R']
    jointname = ['Elbow', 'Shoulder', 'Hip', 'Knee', 'Shoulder_JC', 'Hip_JC', 'Knee_JC']
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
    print(df_list[0])
    joint_angles = table_angle_results(df_list)

    # DOWNLOAD the angle results --> paste into data results
    joint_angles.to_csv(rf'/Users/soowan/Downloads/2024{mmdd}-{p}-{op_games[game_ind]}-angle.csv', encoding = 'utf-8-sig') 

print("\nFOLLOWING FILES DO NOT EXIST:", directory_unknown)
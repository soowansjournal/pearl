'''
---
# **OB1 AutoAnalysis 1-3) Joint Angle ROM**
---

### **_Use for all ANALYSIS since cleaned data is all SAME FORMAT_**

# Steps
# 1) For each game
# 2) For each participant
# 3) For each joint
# 4) For each side
# 5) OP vs MA
# 6) Joint Angle
# 7) Min/Max Angle

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
# op_games = ['PowerR', 'PowerL', 'Wizards', 'War', 'Jet', 'Astro',
#             'BC1', 'BC2', 'BC3', 'BC4', 'BC5', 'BC6', 'BC7', 'BC8', 'BC9']
# ma_games = ['PowerR', 'PowerL', 'Wizards', 'War', 'Jet', 'Astro',
#             'BC1', 'BC2', 'BC3', 'BC4', 'BC5', 'BC6', 'BC7', 'BC8', 'BC9']
op_games = ['PowerR', 'PowerL', 'Wizards', 'War', 'Jet', 'Astro']
ma_games = ['PowerR', 'PowerL', 'Wizards', 'War', 'Jet', 'Astro']


# manually update the boot camp files to corresponding name
boot_camp = ['SeatStarJump',	'HipExt',	'Kick',	'LatStep',	'HipFlex',	'SeatHipFlex',	'StLun',	'BackStep',	'SeatKnExt']
# rename the boot camp exercises
j = 0
for i in range(len(op_games)):
  if 'BC' in op_games[i]:
    op_games[i] = op_games[i] + '-' + boot_camp[j]
    ma_games[i] = ma_games[i] + '-' + boot_camp[j]
    j = j+1


# SELECT FILES HERE
# mmdd_p_all = ['0221_P01', '0314_P02', '0314_P03', '0315_P04', 
#               '0316_P05', '0322_P06', '0402_P07', '0403_P08', '0403_P09', '0404_P10', '0404_P11', 
#               '0406_P12', '0406_P13', '0407_P14', '0407_P15', '0407_P16', '0408_P17', '0408_P18', 
#               '0411_P19', '0412_P20', '0412_P21', '0413_P22', '0420_P23', '0420_P24', '0430_P25', '0502_P26', '0516_P27']
mmdd_p_all = ['0221_P01', '0314_P02', '0314_P03', '0315_P04', 
              '0316_P05', '0322_P06', '0402_P07', '0403_P08', '0403_P09', '0404_P10', '0404_P11', 
              '0406_P12', '0406_P13', '0407_P14', '0407_P15', '0407_P16', '0408_P17', '0408_P18', 
              '0411_P19', '0412_P20', '0412_P21', '0413_P22', '0420_P23', '0420_P24', '0430_P25', '0502_P26', '0516_P27']


# 1) For each game
for game_ind in range(len(op_games)):

    directory_unknown = []
    data = []

    # 2) For each participant
    for mmdd_p in mmdd_p_all:

        print(f'\n\n\n\n\n\n\n\n{mmdd_p}\n\n\n\n\n\n\n\n')
        op_file = '2023' + mmdd_p[:4] + '-' + mmdd_p[-3:] + '-' + op_games[game_ind] + "-Data-OP-CLEAN.csv"
        ma_file = '2023' + mmdd_p[:4] + '-' + mmdd_p[-3:] + '-' + ma_games[game_ind] + "-MA-CLEAN.csv"
        #print(op_file, '\t', ma_file)

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
        
        # 3) For each joint
        # 4) For each side
        # 5) OP vs MA
        # 6) Joint Angle
        # 7) Min/Max Angle

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
                    tmp = pd.DataFrame([single_vals], columns = col, index = [mmdd_p[-3:]])
                    df_list.append(tmp)


        # show results
        joint_angles = table_angle_results(df_list)

        # DOWNLOAD the angle results --> paste into data results
        joint_angles.to_csv(rf'/Users/soowan/Documents/PEARL/Data/Data_OB1/3_Angle/{op_games[game_ind]}/2023{mmdd_p[:4]}-{mmdd_p[-3:]}-{op_games[game_ind]}-angle.csv', encoding = 'utf-8-sig') 

        data.append(joint_angles)

        print("\nFOLLOWING FILES DO NOT EXIST:", directory_unknown)


    joint_angles_overall = pd.concat(data)

    # DOWNLOAD the OVERALL angle results --> paste into data results
    joint_angles_overall.to_csv(rf'/Users/soowan/Documents/PEARL/Data/Data_OB1/3_Angle/{op_games[game_ind]}/2023-{op_games[game_ind]}-angle.csv', encoding = 'utf-8-sig') 
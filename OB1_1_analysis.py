'''
---
# **OB1 AutoAnalysis 1-1) Joint Coordinate Position (R)**
---

### **_Use for all ANALYSIS since cleaned data is all SAME FORMAT_**

# Steps
# 1) For each game
# 2) For each participant
# 3) For each joint
# 4) For each side
# 5) For each coordinate
# 6) OP vs MA
# 7) Joint Data
# 8) Fisher's Z Scores AND Pearson's R Correlation Coefficient

Soowan Choi
'''


from OB1_1_functions import * # todo import other modules


def load_op(op_file):
  # create dataframe from uploaded csv files using pandas.read_csv()
  op = pd.read_csv(op_file) 
  return op 


def load_ma(ma_file):
  # create dataframe from uploaded csv files using pandas.read_csv() & skip the first few rows (3) of information
  ma = pd.read_csv(ma_file) 
  return ma


# Select Game
# op_games = ['PowerR', 'PowerL', 'Wizards', 'War', 'Jet', 'Astro']
# ma_games = ['PowerR', 'PowerL', 'Wizards', 'War', 'Jet', 'Astro']
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
    data_r = []
    data_p = []
    data_z = []

    # 2) For each participant
    for mmdd_p in mmdd_p_all:

        print(f'\n\n\n\n\n\n\n\n{mmdd_p}\n\n\n\n\n\n\n\n')
        op_file = '2023' + mmdd_p[:4] + '-' + mmdd_p[-3:] + '-' + op_games[game_ind] + "-Data-OP-CLEAN.csv"
        ma_file = '2023' + mmdd_p[:4] + '-' + mmdd_p[-3:] + '-' + ma_games[game_ind] + "-MA-CLEAN.csv"


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


        # 1-1) Joint Coordinate Position (R)

        # All Joint Data (39 total)
        op_head = ['Head']
        ma_head = ['Front.Head']
        op_joints = ['Shoulder','Elbow','Wrist','Hip','Knee','Foot']
        ma_joints = ['Shoulder','Elbow','Wrist','ASIS','Knee','Ankle']
        op_side = ['Left','Right']
        ma_side = ['L.','R.']
        xyz = ['Y','Z','X']

        joint_col = []
        joint_corr_z = []
        joint_corr = []
        joint_p_val = []

        # Head Data
        for i in range(len(op_head)):
            for k in range(len(xyz)):
                op_joint = op_head[i] + xyz[k]
                ma_joint = ma_head[i] + xyz[k]
                joint = ma_joint
                if xyz[k] == 'Y':
                    joint, corr, p_val = fiveone(op_final, ma_final, joint, op_joint, ma_joint)  # align horizontal(Y) coordinate
                    joint_corr.append(corr)
                    joint_p_val.append(p_val)
                    joint_col.append(joint)
                    z = z_transform(corr)
                    joint_corr_z.append(z)
                elif xyz[k] == 'Z':
                    joint, corr, p_val = fiveone(op_final, ma_final, joint, op_joint, ma_joint)  # align vertical(Z) coordinate
                    joint_corr.append(corr)
                    joint_p_val.append(p_val)
                    joint_col.append(joint) 
                    z = z_transform(corr)
                    joint_corr_z.append(z)
                elif xyz[k] == 'X':
                    joint, corr, p_val = fiveone(op_final, ma_final, joint, op_joint, ma_joint)  # align depth(X) coordinate
                    joint_corr.append(corr)
                    joint_p_val.append(p_val)
                    joint_col.append(joint)
                    z = z_transform(corr)
                    joint_corr_z.append(z)
        # Body Data
        # 3) For each joint
        for i in range(len(op_joints)):               
            # 4) For each side
            for j in range(len(op_side)):                  
                # 5) For each coordinate
                for k in range(len(xyz)):               
                    op_joint = op_joints[i] + op_side[j] + xyz[k]  # specific OP joint name
                    ma_joint = ma_side[j] + ma_joints[i] + xyz[k]  # specific MA joint name 
                    joint = ma_side[j] + ma_joints[i] + xyz[k]     # joint of interest
                    if xyz[k] == 'Y':
                        # 6) OP vs MA
                        # 7) Joint Data
                        # 8) Fisher's Z Scores AND Pearson's R Correlation Coefficient
                        joint, corr, p_val = fiveone(op_final, ma_final, joint, op_joint, ma_joint)  # align horizontal(Y) coordinate
                        joint_corr.append(corr)
                        joint_p_val.append(p_val)
                        joint_col.append(joint)
                        z = z_transform(corr)
                        joint_corr_z.append(z)
                    elif xyz[k] == 'Z':
                        joint, corr, p_val = fiveone(op_final, ma_final, joint, op_joint, ma_joint)  # align vertical(Z) coordinate
                        joint_corr.append(corr)
                        joint_p_val.append(p_val)
                        joint_col.append(joint)
                        z = z_transform(corr)
                        joint_corr_z.append(z)
                    elif xyz[k] == 'X':
                        joint, corr, p_val = fiveone(op_final, ma_final, joint, op_joint, ma_joint)  # align depth(X) coordinate
                        joint_corr.append(corr)
                        joint_p_val.append(p_val)
                        joint_col.append(joint) 
                        z = z_transform(corr)
                        joint_corr_z.append(z)

        joint_corr = [joint_corr]
        joint_p_val = [joint_p_val]
        joint_corr_z = [joint_corr_z]

        joint_r = pd.DataFrame(joint_corr, columns = joint_col, index = [mmdd_p[-3:]])
        joint_p = pd.DataFrame(joint_p_val, columns = joint_col, index = [mmdd_p[-3:]])
        joint_z = pd.DataFrame(joint_corr_z, columns = joint_col, index = [mmdd_p[-3:]])

        # DOWNLOAD the joint coordinate correlation R-Z-P-Values --> paste into data results
        joint_r.to_csv(rf'/Users/soowan/Documents/PEARL/Data/Data_OB1/1_Coordinate/{op_games[game_ind]}/2023{mmdd_p[:4]}-{mmdd_p[-3:]}-{op_games[game_ind]}-Joint_r.csv', encoding = 'utf-8-sig') 
        joint_p.to_csv(rf'/Users/soowan/Documents/PEARL/Data/Data_OB1/1_Coordinate/{op_games[game_ind]}/2023{mmdd_p[:4]}-{mmdd_p[-3:]}-{op_games[game_ind]}-Joint_p_val.csv', encoding = 'utf-8-sig') 
        joint_z.to_csv(rf'/Users/soowan/Documents/PEARL/Data/Data_OB1/1_Coordinate/{op_games[game_ind]}/2023{mmdd_p[:4]}-{mmdd_p[-3:]}-{op_games[game_ind]}-Joint_z.csv', encoding = 'utf-8-sig') 

        data_r.append(joint_r)
        data_p.append(joint_p)
        data_z.append(joint_z)
        
        print("\nFOLLOWING FILES DO NOT EXIST:", directory_unknown)

    joint_r_overall = pd.concat(data_r)
    joint_p_overall = pd.concat(data_p)
    joint_z_overall = pd.concat(data_z)

    # DOWNLOAD the OVERALL joint coordinate correlation R-Z-P-Values --> paste into data results
    joint_r_overall.to_csv(rf'/Users/soowan/Documents/PEARL/Data/Data_OB1/1_Coordinate/{op_games[game_ind]}/2023-{op_games[game_ind]}-Joint_r.csv', encoding = 'utf-8-sig')
    joint_p_overall.to_csv(rf'/Users/soowan/Documents/PEARL/Data/Data_OB1/1_Coordinate/{op_games[game_ind]}/2023-{op_games[game_ind]}-Joint_p_val.csv', encoding = 'utf-8-sig')
    joint_z_overall.to_csv(rf'/Users/soowan/Documents/PEARL/Data/Data_OB1/1_Coordinate/{op_games[game_ind]}/2023-{op_games[game_ind]}-Joint_z.csv', encoding = 'utf-8-sig')
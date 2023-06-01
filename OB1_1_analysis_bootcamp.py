'''
---
# **OB1 AutoAnalysis 1-1) Joint Coordinate Position (R)**
---

### **_Use for all ANALYSIS since cleaned data is all SAME FORMAT_**

- Same as OB1_1_analysis but for Bootle Boot Camp

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
# mmdd_p_all = ['0221_P01', '0314_P02', '0314_P03', '0315_P04', 
#               '0316_P05', '0322_P06', '0402_P07', '0403_P08', '0403_P09', '0404_P10', '0404_P11', 
#               '0406_P12', '0406_P13', '0407_P14', '0407_P15', '0407_P16', '0408_P17', '0408_P18', 
#               '0411_P19', '0412_P20', '0412_P21', '0413_P22', '0420_P23', '0420_P24', '0430_P25', '0502_P26', '0516_P27']
mmdd_p_all = ['0221_P01', '0314_P02', '0314_P03', '0315_P04', 
              '0316_P05', '0322_P06', '0402_P07', '0403_P08', '0403_P09', '0404_P10', '0404_P11', 
              '0406_P12', '0406_P13', '0407_P14', '0407_P15', '0407_P16', '0408_P17', '0408_P18', 
              '0411_P19', '0412_P20', '0412_P21', '0413_P22', '0420_P23', '0420_P24', '0430_P25', '0502_P26', '0516_P27']


strength_r = []
strength_p = []
strength_z = []

cardio_r = []
cardio_p = []
cardio_z = []

seated_r = []
seated_p = []
seated_z = []

static_r = []
static_p = []
static_z = []


# 1) For each game
for game_ind in range(len(op_games)):

    directory_unknown = []
    data_r = []
    data_p = []
    data_z = []

    # 2) For each participant
    for mmdd_p in mmdd_p_all:

        print(f'\n\n\n\n\n\n\n\n{mmdd_p}\n\n\n\n\n\n\n\n')

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

        array_error = []

        try:

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
        except:
            # if array contains INFs or NaNs go to next joint
            array_error.append(op_joint)
            continue

        joint_corr = [joint_corr]
        joint_p_val = [joint_p_val]
        joint_corr_z = [joint_corr_z]

        joint_r = pd.DataFrame(joint_corr, columns = joint_col, index = [mmdd_p[-3:]])
        joint_p = pd.DataFrame(joint_p_val, columns = joint_col, index = [mmdd_p[-3:]])
        joint_z = pd.DataFrame(joint_corr_z, columns = joint_col, index = [mmdd_p[-3:]])

        # DOWNLOAD EACH GAME EACH PARTICIPANT
        joint_r.to_csv(rf'/Users/soowan/Documents/PEARL/Data/Data_OB1/1_Coordinate/BC_Boot_Camp/{op_games[game_ind]}/2023{mmdd_p[:4]}-{mmdd_p[-3:]}-{op_games[game_ind]}-Joint_r.csv', encoding = 'utf-8-sig') 
        joint_p.to_csv(rf'/Users/soowan/Documents/PEARL/Data/Data_OB1/1_Coordinate/BC_Boot_Camp/{op_games[game_ind]}/2023{mmdd_p[:4]}-{mmdd_p[-3:]}-{op_games[game_ind]}-Joint_p_val.csv', encoding = 'utf-8-sig') 
        joint_z.to_csv(rf'/Users/soowan/Documents/PEARL/Data/Data_OB1/1_Coordinate/BC_Boot_Camp/{op_games[game_ind]}/2023{mmdd_p[:4]}-{mmdd_p[-3:]}-{op_games[game_ind]}-Joint_z.csv', encoding = 'utf-8-sig') 


        data_r.append(joint_r)
        data_p.append(joint_p)
        data_z.append(joint_z)
        
        print("\nFOLLOWING FILES DO NOT EXIST:", directory_unknown)


    joint_r_overall = pd.concat(data_r)
    joint_p_overall = pd.concat(data_p)
    joint_z_overall = pd.concat(data_z)

    # DOWNLOAD EACH GAME ALL PARTICIPANT
    joint_r_overall.to_csv(rf'/Users/soowan/Documents/PEARL/Data/Data_OB1/1_Coordinate/BC_Boot_Camp/{op_games[game_ind]}/2023-{op_games[game_ind]}-Joint_r.csv', encoding = 'utf-8-sig')
    joint_p_overall.to_csv(rf'/Users/soowan/Documents/PEARL/Data/Data_OB1/1_Coordinate/BC_Boot_Camp/{op_games[game_ind]}/2023-{op_games[game_ind]}-Joint_p_val.csv', encoding = 'utf-8-sig')
    joint_z_overall.to_csv(rf'/Users/soowan/Documents/PEARL/Data/Data_OB1/1_Coordinate/BC_Boot_Camp/{op_games[game_ind]}/2023-{op_games[game_ind]}-Joint_z.csv', encoding = 'utf-8-sig')




    # Group Boot Camp into Four Categories
    # STRENGTH
    strength = ['Sqt', 'StLun', 'VMODip', 'HipFlex', 'HipExt', 'HipAbd', 'Kick', 'LatStep', 'BackStep']
    strength = ['Sqt', 'StLun', 'VMODip', 'HipFlex', 'HipExt', 'HipAbd', 'Kick', 'LatStep', 'BackStep']
    # CARDIO
    cardio = ['StarJump', 'Run']
    cardio = ['StarJump', 'Run']
    # SEATED
    seated = ['SeatKnExt', 'SeatHipFlex', 'SeatStarJump']
    seated = ['SeatKnExt', 'SeatHipFlex', 'SeatStarJump']
    # STATIC
    static = ['SeatClfStr', 'ForStep', 'CalfStr', 'TdemStnce']
    static = ['SeatClfStr', 'ForStep', 'CalfStr', 'TdemStnce']


    if op_games[game_ind] in strength:
        # Insert the new column at the first index location
        joint_r_overall.insert(0, 'Game', np.repeat(op_games[game_ind], len(joint_r_overall)))
        strength_r.append(joint_r_overall)

        joint_p_overall.insert(0, 'Game', np.repeat(op_games[game_ind], len(joint_p_overall)))
        strength_p.append(joint_p_overall)

        joint_z_overall.insert(0, 'Game', np.repeat(op_games[game_ind], len(joint_z_overall)))
        strength_z.append(joint_z_overall)

    elif op_games[game_ind] in cardio:
        joint_r_overall.insert(0, 'Game', np.repeat(op_games[game_ind], len(joint_r_overall)))
        cardio_r.append(joint_r_overall)

        joint_p_overall.insert(0, 'Game', np.repeat(op_games[game_ind], len(joint_p_overall)))
        cardio_p.append(joint_p_overall)

        joint_z_overall.insert(0, 'Game', np.repeat(op_games[game_ind], len(joint_z_overall)))
        cardio_z.append(joint_z_overall)

    elif op_games[game_ind] in seated:
        joint_r_overall.insert(0, 'Game', np.repeat(op_games[game_ind], len(joint_r_overall)))
        seated_r.append(joint_r_overall)

        joint_p_overall.insert(0, 'Game', np.repeat(op_games[game_ind], len(joint_p_overall)))
        seated_p.append(joint_p_overall)

        joint_z_overall.insert(0, 'Game', np.repeat(op_games[game_ind], len(joint_z_overall)))
        seated_z.append(joint_z_overall)

    elif op_games[game_ind] in static:
        joint_r_overall.insert(0, 'Game', np.repeat(op_games[game_ind], len(joint_r_overall)))
        static_r.append(joint_r_overall)

        joint_p_overall.insert(0, 'Game', np.repeat(op_games[game_ind], len(joint_p_overall)))
        static_p.append(joint_p_overall)

        joint_z_overall.insert(0, 'Game', np.repeat(op_games[game_ind], len(joint_z_overall)))
        static_z.append(joint_z_overall)


joint_r_strength = pd.concat(strength_r)
joint_r_cardio = pd.concat(cardio_r)
joint_r_seated = pd.concat(seated_r)
joint_r_static = pd.concat(static_r)

joint_p_strength = pd.concat(strength_p)
joint_p_cardio = pd.concat(cardio_p)
joint_p_seated = pd.concat(seated_p)
joint_p_static = pd.concat(static_p)

joint_z_strength = pd.concat(strength_z)
joint_z_cardio = pd.concat(cardio_z)
joint_z_seated = pd.concat(seated_z)
joint_z_static = pd.concat(static_z)


# DOWNLOAD GROUPED GAMES ALL PARTICIPANT 
joint_r_strength.to_csv(rf'/Users/soowan/Documents/PEARL/Data/Data_OB1/1_Coordinate/BC_Strength/2023-STRENGTH-Joint_r.csv', encoding = 'utf-8-sig')
joint_r_cardio.to_csv(rf'/Users/soowan/Documents/PEARL/Data/Data_OB1/1_Coordinate/BC_Cardio/2023-CARDIO-Joint_r.csv', encoding = 'utf-8-sig')
joint_r_seated.to_csv(rf'/Users/soowan/Documents/PEARL/Data/Data_OB1/1_Coordinate/BC_Seated/2023-SEATED-Joint_r.csv', encoding = 'utf-8-sig')
joint_r_static.to_csv(rf'/Users/soowan/Documents/PEARL/Data/Data_OB1/1_Coordinate/BC_Static/2023-STATIC-Joint_r.csv', encoding = 'utf-8-sig')

joint_p_strength.to_csv(rf'/Users/soowan/Documents/PEARL/Data/Data_OB1/1_Coordinate/BC_Strength/2023-STRENGTH-Joint_p_val.csv', encoding = 'utf-8-sig')
joint_p_cardio.to_csv(rf'/Users/soowan/Documents/PEARL/Data/Data_OB1/1_Coordinate/BC_Cardio/2023-CARDIO-Joint_p_val.csv', encoding = 'utf-8-sig')
joint_p_seated.to_csv(rf'/Users/soowan/Documents/PEARL/Data/Data_OB1/1_Coordinate/BC_Seated/2023-SEATED-Joint_p_val.csv', encoding = 'utf-8-sig')
joint_p_static.to_csv(rf'/Users/soowan/Documents/PEARL/Data/Data_OB1/1_Coordinate/BC_Static/2023-STATIC-Joint_p_val.csv', encoding = 'utf-8-sig')

joint_z_strength.to_csv(rf'/Users/soowan/Documents/PEARL/Data/Data_OB1/1_Coordinate/BC_Strength/2023-STRENGTH-Joint_z.csv', encoding = 'utf-8-sig')
joint_z_cardio.to_csv(rf'/Users/soowan/Documents/PEARL/Data/Data_OB1/1_Coordinate/BC_Cardio/2023-CARDIO-Joint_z.csv', encoding = 'utf-8-sig')
joint_z_seated.to_csv(rf'/Users/soowan/Documents/PEARL/Data/Data_OB1/1_Coordinate/BC_Seated/2023-SEATED-Joint_z.csv', encoding = 'utf-8-sig')
joint_z_static.to_csv(rf'/Users/soowan/Documents/PEARL/Data/Data_OB1/1_Coordinate/BC_Static/2023-STATIC-Joint_z.csv', encoding = 'utf-8-sig')
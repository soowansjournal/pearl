'''
---
# **OB1 AutoAnalysis 1-2) Body Segment Length**
---

### **_Use for all ANALYSIS since cleaned data is all SAME FORMAT_**

# Steps
# 1) For each game
# 2) For each participant
# 3) For each segment
# 4) For each side
# 5) OP vs MA
# 6) Body Segment
# 7) Coefficient of Variation

Soowan Choi
'''


from OB1_2_functions import * # todo import other modules


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


        # 1-2) Body Segment Length (CV)
        
        # 3) For each segment
        # 4) For each side
        # 5) OP vs MA
        # 6) Body Segment
        # 7) Coefficient of Variation
        op_cov = fivetwo(op_final, ma_final, [mmdd_p[-3:]])

        # DOWNLOAD the COV results --> paste into data results
        op_cov.to_csv(rf'/Users/soowan/Documents/PEARL/Data/Data_OB1/2_Segment/{op_games[game_ind]}/2023{mmdd_p[:4]}-{mmdd_p[-3:]}-{op_games[game_ind]}-cov.csv', encoding = 'utf-8-sig') 

        data.append(op_cov)
        
        print("\nFOLLOWING FILES DO NOT EXIST:", directory_unknown)

    op_cov_overall = pd.concat(data)

    # DOWNLOAD the OVERALL COV results --> paste into data results
    op_cov_overall.to_csv(rf'/Users/soowan/Documents/PEARL/Data/Data_OB1/2_Segment/{op_games[game_ind]}/2023-{op_games[game_ind]}-cov.csv', encoding = 'utf-8-sig') 
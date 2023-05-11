'''
---
# **OB1 AutoAnalysis 1-1) Joint Coordinate Position (R)**
---

### **_Use for all ANALYSIS since cleaned data is all SAME FORMAT_**

Soowan Choi
'''


from OB1_autoanalysis_1_functions import * # todo import other modules


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

op_games = ['PowerR', 'PowerL', 'Wizards', 'War', 'Jet', 'Astro',
            'BC1', 'BC2', 'BC3', 'BC4', 'BC5', 'BC6', 'BC7', 'BC8', 'BC9']

ma_games = ['PowerR', 'PowerL', 'Wizards', 'War', 'Jet', 'Astro',
            'BC1', 'BC2', 'BC3', 'BC4', 'BC5', 'BC6', 'BC7', 'BC8', 'BC9']

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
mmdd = '0314' 
p = 'P03'
mmdd_p = mmdd + '_' + p

for game_ind in range(len(op_games)):
  print(f'\n\n\n\n\n\n\n\n{op_games[game_ind]}\n\n\n\n\n\n\n\n')
  op_file = '2023' + mmdd + '-' + p + '-' + op_games[game_ind] + "-Data-OP-CLEAN.csv"
  ma_file = '2023' + mmdd + '-' + p + '-' + ma_games[game_ind] + "-MA-CLEAN.csv"
  #print(op_file, '\t', ma_file)


  # Load OP Data
  op = load_op('/Users/soowan/Documents/PEARL/Data/Data_0551/2023_' + mmdd_p + '/Clean_' + mmdd_p + '/' + op_file)
  print(op.head(3))

  # Load MA Data
  ma = load_ma('/Users/soowan/Documents/PEARL/Data/Data_0551/2023_' + mmdd_p + '/Clean_' + mmdd_p + '/' + ma_file)
  print(ma.head(3))


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

  # Head Data
  for i in range(len(op_head)):
    for k in range(len(xyz)):
      op_joint = op_head[i] + xyz[k]
      ma_joint = ma_head[i] + xyz[k]
      joint = ma_joint
      if xyz[k] == 'Y':
        joint, corr = fiveone(op_final, ma_final, joint, op_joint, ma_joint)  # align horizontal(Y) coordinate
        joint_col.append(joint)
        z = z_transform(corr)
        joint_corr_z.append(z)
      elif xyz[k] == 'Z':
        joint, corr = fiveone(op_final, ma_final, joint, op_joint, ma_joint)  # align vertical(Z) coordinate
        joint_col.append(joint) 
        z = z_transform(corr)
        joint_corr_z.append(z)
      elif xyz[k] == 'X':
        joint, corr = fiveone(op_final, ma_final, joint, op_joint, ma_joint)  # align depth(X) coordinate
        joint_col.append(joint)
        z = z_transform(corr)
        joint_corr_z.append(z)
  # Body Data
  for i in range(len(op_joints)):                   # for each joints
    for j in range(len(op_side)):                   # for each sides 
      for k in range(len(xyz)):                     # for each xyz 
        op_joint = op_joints[i] + op_side[j] + xyz[k]  # specific OP joint name
        ma_joint = ma_side[j] + ma_joints[i] + xyz[k]  # specific MA joint name 
        joint = ma_side[j] + ma_joints[i] + xyz[k]     # joint of interest
        if xyz[k] == 'Y':
          joint, corr = fiveone(op_final, ma_final, joint, op_joint, ma_joint)  # align horizontal(Y) coordinate
          joint_col.append(joint)
          z = z_transform(corr)
          joint_corr_z.append(z)
        elif xyz[k] == 'Z':
          joint, corr = fiveone(op_final, ma_final, joint, op_joint, ma_joint)  # align vertical(Z) coordinate
          joint_col.append(joint)
          z = z_transform(corr)
          joint_corr_z.append(z)
        elif xyz[k] == 'X':
          joint, corr = fiveone(op_final, ma_final, joint, op_joint, ma_joint)  # align depth(X) coordinate
          joint_col.append(joint) 
          z = z_transform(corr)
          joint_corr_z.append(z)

  joint_corr_z = [joint_corr_z]

  joint_z = pd.DataFrame(joint_corr_z, columns = joint_col)

  # DOWNLOAD the joint coordinate correlation Z-Values --> paste into data results
  joint_z.to_csv(rf'/Users/soowan/Downloads/2023{mmdd}-{p}-{op_games[game_ind]}-Joint_z.csv', encoding = 'utf-8-sig') 
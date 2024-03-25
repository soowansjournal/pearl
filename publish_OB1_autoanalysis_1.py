'''
---
# **OB1 AutoAnalysis 1-1) Joint Coordinate Position (R)**
---

### **_Use for all ANALYSIS since cleaned data is all SAME FORMAT_**

Soowan Choi
'''


from publish_OB1_autoanalysis_1_functions import * # todo import other modules


def load_op(op_file):
  # create dataframe from uploaded csv files using pandas.read_csv()
  op = pd.read_csv(op_file) 
  return op 


def load_ma(ma_file):
  # create dataframe from uploaded csv files using pandas.read_csv() & skip the first few rows (3) of information
  ma = pd.read_csv(ma_file) 
  return ma


# Select Game --> Bootle Blast
# op_games = ['PowerR', 'PowerL', 'Wizards', 'War', 'Jet', 'Astro']
# ma_games = ['PowerR', 'PowerL', 'Wizards', 'War', 'Jet', 'Astro']
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
      data_vis(op_align_joints, ma_align_joints, joint, op_joint, ma_joint)  # align depth(X) coordinate

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


  # 1-1) Joint Coordinate Position (R)

  # All Joint Data (39 total)
  op_head = ['Head']
  ma_head = ['Front.Head']
  # op_joints = ['Shoulder','Elbow','Wrist','Hip','Knee','Foot']
  op_joints = ['Shoulder','Elbow','Wrist','Hip','Knee','Foot','Hip','Knee','Foot']
  # ma_joints = ['Shoulder','Elbow','Wrist','ASIS','Knee','Ankle']
  ma_joints = ['Shoulder','Elbow','Wrist','ASIS','Knee','Ankle','Hip_JC','Knee_JC','Ankle_JC']
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
  for i in range(len(op_joints)):                   # for each joints
    for j in range(len(op_side)):                   # for each sides 
      for k in range(len(xyz)):                     # for each xyz 
        op_joint = op_joints[i] + op_side[j] + xyz[k]  # specific OP joint name

        if "_JC" in ma_joints[i]:
          ma_joint = "V_" + ma_side[j] + ma_joints[i] + xyz[k]  # specific MA joint name 
          joint = "V_" + ma_side[j] + ma_joints[i] + xyz[k]     # joint of interest
        else:
          ma_joint = ma_side[j] + ma_joints[i] + xyz[k]  # specific MA joint name 
          joint = ma_side[j] + ma_joints[i] + xyz[k]     # joint of interest

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

  joint_corr = [joint_corr]
  joint_p_val = [joint_p_val]
  joint_corr_z = [joint_corr_z]

  joint_r = pd.DataFrame(joint_corr, columns = joint_col, index = [mmdd_p[-3:]])
  joint_p = pd.DataFrame(joint_p_val, columns = joint_col, index = [mmdd_p[-3:]])
  joint_z = pd.DataFrame(joint_corr_z, columns = joint_col, index = [mmdd_p[-3:]])

  # DOWNLOAD the joint coordinate correlation Z-R-PValues --> paste into data results
  joint_r.to_csv(rf'/Users/soowan/Downloads/2024{mmdd}-{p}-{op_games[game_ind]}-Joint_r.csv', encoding = 'utf-8-sig')
  joint_p.to_csv(rf'/Users/soowan/Downloads/2024{mmdd}-{p}-{op_games[game_ind]}-Joint_p.csv', encoding = 'utf-8-sig')
  joint_z.to_csv(rf'/Users/soowan/Downloads/2024{mmdd}-{p}-{op_games[game_ind]}-Joint_z.csv', encoding = 'utf-8-sig') 

print("\nFOLLOWING FILES DO NOT EXIST:", directory_unknown)
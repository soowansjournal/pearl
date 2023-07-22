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


def filter(data_filte, data_hz, cutoff):
  # save column names to rename new filtered data
  col_names = list(data_filte.columns)              
  time = data_filte['Time'].reset_index()
  del time['index']

  empty = []                                 
  # filter all but the first time column and store filtered array data in a list
  for col in data_filte.columns[1:]:                  
    y = data_filte[col]                          # for each column data
    y2 = butter_lowpass_filter(y, cutoff, data_hz, 6)    # filter the column data (data, 1-3-6-9-12 Hz cutoff, fs, 6th order)
    empty.append(y2)                             # store filtered data

  # create dataframe from stored filtered data & rename columns & reinsert time column
  data_filte = pd.DataFrame(empty).T                   
  data_filte.columns = col_names[1:]                 
  data_filte.insert(0,'Time',time)    

  return data_filte    


def coord_op(op_coord):
  '2.2.2) Coordinate Transformation (Y ← OP_X | Z ← OP_Y | X ← OP_Z) + (Y ← MA_X | Z ← MA_Z | X ← negMA_Y)'

  print('\n2.2.2) COORDINATE TRANSFORMATION (Y ← OP_X | Z ← OP_Y | X ← OP_Z) + (Y ← MA_X | Z ← MA_Z | X ← negMA_Y)\n----------------------------\n')

  # remove the X Y Z indicators from column names
  op_coord.columns = op_coord.columns.str.replace('X$','',regex =True)
  op_coord.columns = op_coord.columns.str.replace('Y$','',regex =True)
  op_coord.columns = op_coord.columns.str.replace('Z$','',regex =True)

  # create list of new column names
  naming_system = ['Y','Z','X']  # list the specific column measurement names to add to new column
  op_tmp = op_coord.columns.copy()     # create a copy of the column name order
  new_col = []                   # empty list to store new column names
  for i in range(1, 57, 3):      # loop from 'Head' index to index of last relevant column, increment every 3 steps identical to csv file
    for j in range(0,3):
      op_coord.rename(columns = {op_coord.columns[i+j]:op_tmp[i+j] + naming_system[j]}, inplace= True)    # change the specific name of the six succeeding columns in place
      new_col.append(op_coord.columns[i])

  # list of new column names  
  op_col = list(op_coord.columns)   # convert column names to a list (lists are mutable)
  for i in range(1,57):       # change relevant column names
    op_col[i] = new_col[i-1]  # change the column name using index

  # rename columns in data using list of new column names
  op_coord.columns = op_col

  return op_coord


def coord_ma(ma_coord, pre_or_post):
  '3.2.2) Coordinate Transformation'

  print('\n3.2.2) COORDINATE TRANSFORMATION\n----------------------------\n')

  # isolate for relevant column names 
  new_names = []             
  for name in ma_coord.columns:
    if 'Unnamed' in name:     # irrelevant columns
      pass                 
    else:                     # relevant columns     
      new_names.append(name) 

  # search if relevant column exists in the dataframe and locate the index 
  dic = {}                                                     
  i = 0                                                             
  for name in new_names: 
    if name in ma_coord.columns:
      dic[i] = [name, 'index location:', ma_coord.columns.get_loc(name)]  # add index location of the column name to dictionary
      i += 1                                                        # increment dictionary index

  # rename columns of three succeeding index (to same name) from new_names list...
  if pre_or_post == "pre0316":
     # We changed OP to MA
     naming_system = ['X','Y','Z']  
  elif pre_or_post == "post0316":
     # We changed both OP and MA separately
     naming_system = ['Y','X','Z']    
  # get the index of the starting column name, which is 'Front.Head'
  start = ma_coord.columns.get_loc('Front.Head')            
  # create a copy of the column names
  ma_tmp = ma_coord.columns.copy()                    
  # loop from index of 'Front.Head' to index of last relevant column, increment every 3 steps (identical to csv file)    
  for i in range(start, dic[max(dic)][-1], 3):  
    # loop through the next three succeeding index
    for j in range(0,3):                            
      # change the specific name of the six succeeding columns in place
      ma_coord.rename(columns = {ma_coord.columns[i+j]:ma_tmp[i] + naming_system[j]}, inplace= True)

  return ma_coord  


def nullv_ma(ma_nullv):
  '3.2.3) INTERPOLATE NULL'

  print('\n3.2.3) INTERPOLATE NULL\n----------------------------\n')

  # columns with NULL values BEFORE INTERPOLATING 
  nulls = []
  print('Joints with NULL values BEFORE INTERPOLATING NULL:')
  for col in ma_nullv.columns:
    null = ma_nullv[col].isnull().unique()
    if len(null) > 1 or True in null:
      nulls.append(col)
  print(nulls)

  # Important columns with NULL values BEFORE INTERPOLATING
  isna = []
  important = ['Front.Head','.Shoulder','.Elbow','.Wrist','.ASIS','.Knee','.Ankle']
  print('\nIMPORTANT Joints with NULL values BEFORE INTERPOLATING NULL:')
  for i in range(len(nulls)):
    for joint in important:
      if joint in nulls[i]:
        isna.append(nulls[i])
        isna.append(nulls[i+1])
        isna.append(nulls[i+2])
  print(isna)
  print('\n')

  # count how many NULL values to clean in each important column
  for col in isna:
    if "Unnamed" not in col:
      #print(f"NULL Value Locations of the {col} Column:")
      count = 0
      for i,value in enumerate(list(ma_nullv[col])):
        if str(value) == 'nan':
          #print(i,value)
          count += 1
      print(f"{count} NULL Values in the {col} column")

  # find index with null values 
  null_index = []
  for col in isna:
    if "Unnamed" not in col:
      for index, val in enumerate(ma_nullv[col]):
        if str(val) == 'nan':
          null_index.append(index)

  # check if null appears at middle of data (instead of beginning or end)
  for i in range(len(null_index) - 1):
    if (null_index[i+1] - null_index[i]) != 1:
      print(f"Null Appears Middle of Data: {null_index[i-3:i+3]}")

  # interpolate and fill null values 
  for col in isna:    
    ma_nullv[col] = ma_nullv[col].astype(float)
    ma_nullv[col] = ma_nullv[col].interpolate(method = 'polynomial', order = 2, limit_direction = 'forward')
    ma_nullv[col] = ma_nullv[col].astype(str)
  
  # ***backwards fill AFTER initial interpolation, this data will be REMOVED later but required now as placeholders***
  for col in isna:    
    ma_nullv[col] = ma_nullv[col].astype(float)
    ma_nullv[col] = ma_nullv[col].interpolate(method = 'linear', limit_direction = 'backward')
    ma_nullv[col] = ma_nullv[col].astype(str)
  # Interpolate remaining missing values near end of dataframe
  for col in isna:    
    ma_nullv[col] = ma_nullv[col].astype(float)
    ma_nullv[col] = ma_nullv[col].interpolate(method = 'linear')
    ma_nullv[col] = ma_nullv[col].astype(str)

  # columns with NULL values AFTER INTERPOLATING
  nulls = []
  print('Joints with NULL values AFTER INTERPOLATING NULL:')
  for col in ma_nullv.columns:
    null = ma_nullv[col].isnull().unique()
    if len(null) > 1:
      nulls.append(col)
  print(nulls)

  # IMPORTANT columns with NULL values AFTER INTERPOLATING 
  isna = []
  important = ['.Shoulder','.Elbow','.Wrist','.ASIS','.Knee','.Ankle']
  print('\nIMPORTANT Joints with NULL values AFTER INTERPOLATING NULL:')
  for col in nulls:
    for joint in important:
      if joint in col:
        isna.append(col)
  print(isna)
  print('\n')

  # count how many NULL values to clean in each important column
  for col in isna:
    #print(f"NULL Value Locations of the {col} Column:")
    count = 0
    for i,value in enumerate(list(ma_nullv[col])):
      if str(value) == 'nan':
        #print(i,value)
        # Drop null values at end of frames
        ma_nullv = ma_nullv.drop(ma_nullv.index[i])
        count += 1
    print(f"{count} NULL Values in the {col} column")



  return ma_nullv


def resam_ma(ma_resam, op_synch):
  # to resampled data
  from scipy import signal
  from sklearn.utils import resample
  '3.2.5) Resample (30 Hz)'

  print('\n3.2.5) RESAMPLE (30 HZ)\n----------------------------\n')

  print(f'The total time that motion runs is {ma_resam.Time.iloc[-1] - ma_resam.Time.iloc[0]} secs \n')
  print(f'Shape of Motion BEFORE resampling (downsampling to 30 Hz): {ma_resam.shape}')

  # save column and row to recreate dataframe
  ma_col = ma_resam.columns
  ma_row = ma_resam.index

  # resample motion dataframe
  secs = (ma_resam.Time.iloc[-1] - ma_resam.Time.iloc[0])
  samps = op_synch.shape[0] #int(secs*30)                # number of samples to resample
  ma_resam = signal.resample(ma_resam, samps)            # array of resampled data
  ma_resam = pd.DataFrame(ma_resam, columns = ma_col)    # recreate the dataframe
  print(f'Shape of Motion AFTER resampling (downsampling to 30 Hz): {ma_resam.shape} \n')

  # check which columns have null values from the resampled data 
  print('Columns with null values AFTER RESAMPLING:')
  for col in ma_resam.columns:
    null = str(ma_resam[col].isnull().unique()[0])
    if null == 'True':
      print('',col)  

  return ma_resam


def load_op(op_file):
  # create dataframe from uploaded csv files using pandas.read_csv()
  op = pd.read_csv(op_file) 
  return op 


def load_ma(ma_file):
  # create dataframe from uploaded csv files using pandas.read_csv() & skip the first few rows (3) of information
  ma = pd.read_csv(ma_file) 
  return ma


# Select Game
# op_games = ['Pediatric', 'SingleL', 'SingleR', 'Five', 'Thirty']
# ma_games = ['Pediatric', 'SingleL', 'SingleR', 'Five', 'Thirty']
op_games = ['Pediatric', 'SingleL', 'SingleR', 'Five', 'Thirty']
ma_games = ['Pediatric', 'SingleL', 'SingleR', 'Five', 'Thirty']
 

# SELECT FILES HERE
# mmdd_p_all = ['0221_P01', '0314_P02', '0314_P03', '0315_P04', 
#               '0316_P05', '0322_P06', '0402_P07', '0403_P08', '0403_P09', '0404_P10', '0404_P11', 
#               '0406_P12', '0406_P13', '0407_P14', '0407_P15', '0407_P16', '0408_P17', '0408_P18', 
#               '0411_P19', '0412_P20', '0412_P21', '0413_P22', '0420_P23', '0420_P24', '0430_P25', '0502_P26', '0516_P27', '0601_P28']
mmdd_p_all = ['0221_P01', '0314_P02', '0314_P03', '0315_P04', 
              '0316_P05', '0322_P06', '0402_P07', '0403_P08', '0403_P09', '0404_P10', '0404_P11', 
              '0406_P12', '0406_P13', '0407_P14', '0407_P15', '0407_P16', '0408_P17', '0408_P18', 
              '0411_P19', '0412_P20', '0412_P21', '0413_P22', '0420_P23', '0420_P24', '0430_P25', '0502_P26', '0516_P27', '0601_P28']


# 1) For each game
for game_ind in range(len(op_games)):

    directory_unknown = []
    data_r = []
    data_p = []
    data_z = []

    # 2) For each participant
    for mmdd_p in mmdd_p_all:

        print(f'\n\n\n\n\n\n\n\n{mmdd_p}\n\n\n\n\n\n\n\n')
        # Single Leg Stance Test
        if 'Single' in op_games[game_ind]:
            op_file = '2023' + mmdd_p[:4] + '-' + mmdd_p[-3:] + '-' + 'BC-SLS' + "-Data-OP-CLEAN.csv"
            ma_file = '2023' + mmdd_p[:4] + '-' + mmdd_p[-3:] + '-' + 'BC-SLS' + "-MA-CLEAN.csv"
        # Sit-to-Stand Test
        elif op_games[game_ind] == 'Five' or op_games[game_ind] == 'Thirty':
            op_file = '2023' + mmdd_p[:4] + '-' + mmdd_p[-3:] + '-' + 'BC-StS' + "-Data-OP-CLEAN.csv"
            ma_file = '2023' + mmdd_p[:4] + '-' + mmdd_p[-3:] + '-' + 'BC-StS' + "-MA-CLEAN.csv"
        # Pediatric Reach Test
        elif op_games[game_ind] == 'Pediatric':
            op_file = '2023' + mmdd_p[:4] + '-' + mmdd_p[-3:] + '-' + 'Pediatric' + "-Data-OP-CLEAN.csv"
            ma_file = '2023' + mmdd_p[:4] + '-' + mmdd_p[-3:] + '-' + 'Pediatric' + "-MA-CLEAN.csv"


        if op_games[game_ind] == 'SingleL':
            path = '/Users/soowan/Documents/PEARL/Data/Data_OB2/Clean_SCA_SLS/SingleL/'
        elif op_games[game_ind] == 'SingleR':
            path = '/Users/soowan/Documents/PEARL/Data/Data_OB2/Clean_SCA_SLS/SingleR/'
        elif op_games[game_ind] == 'Five':
            path = '/Users/soowan/Documents/PEARL/Data/Data_OB2/Clean_SCA_STS/Five/'
        elif op_games[game_ind] == 'Thirty':
            path = '/Users/soowan/Documents/PEARL/Data/Data_OB2/Clean_SCA_STS/Thirty/'
        elif op_games[game_ind] == 'Pediatric':
            path = '/Users/soowan/Documents/PEARL/Data/Data_0551/2023_' + mmdd_p + '/Auto_Clean_' + mmdd_p + '/'

        try:
            op = load_op(path + op_file)
            ma = load_ma(path + ma_file)

        except FileNotFoundError:
            # if directory game file doesn't exist, go to next game
            directory_unknown.append(op_file)
            continue


        # need to CLEAN (fix column names/resample) for all assessments except for pediatric
        if op_games[game_ind] != 'Pediatric':
            op_final = op.copy() 
            ma_final = ma.copy()   
            
            # Coordinate Transformation (Translation): Rename Column Names
            # Post0316
            if mmdd_p[-3:] != 'P01' and mmdd_p[-3:] != 'P02' and mmdd_p[-3:] != 'P03' and mmdd_p[-3:] != 'P04':
                ma_final = coord_ma(ma_final, 'post0316')
                # MA depth coordinate is negative of OP depth coordinate
                for col in range(len(ma_final.columns)):
                   if 'X' in ma_final.columns[col]:
                      ma_final.iloc[:,col] = -1* ma_final.iloc[:,col]
            # Pre0316
            else:
                ma_final = coord_ma(ma_final, 'pre0316')
                      
            
            op_final = coord_op(op_final)

            # Interpolate Missing Values in MA
            ma_final = nullv_ma(ma_final)

            # Resample MA to Match OP --> Final Results
            ma_final = resam_ma(ma_final, op_final)

            # Convert OP (m --> cm) and MA values (mm --> cm) to cm
            op_final = op_final.astype(float)
            op_final.iloc[:,1:57] = op_final.iloc[:,1:57] *100
            ma_final = ma_final.astype(float)
            ma_final.iloc[:,2:] = ma_final.iloc[:,2:] / 10

            # Convert OP time values to seconds and set as MA time value
            op_final = op_final.reset_index().drop(columns=['index'])
            op_final.iloc[:,0] = op_final.iloc[:, 0] / 1000
            ma_final = ma_final.reset_index().drop(columns=['index'])
            ma_final.Time = op_final.iloc[:, 0] 

            # Filter OP ONLY (MA Filtered Using Cortex Software)
            # OP Frequency
            op_frames = len(op_final.Time)
            op_seconds = (op_final.Time.iloc[-1] - op_final.Time.iloc[0])  
            op_hz = op_frames / op_seconds
            op_final = filter(op_final, op_hz, 3)  # Choose Cut-Off Frequency

        # If game is "Pediatric" it is already cleaned
        else:
           op_final = op.copy()
           ma_final = ma.copy()

        op_align_joints, ma_align_joints = op_final, ma_final



        


        # # Visualize ALL Data (39 graphs total)
        # op_head = ['Head']
        # ma_head = ['Front.Head']
        # op_side = ['Left','Right']
        # ma_side = ['L.','R.']
        # xyz = ['Y','Z','X']
        # # Head Data
        # for i in range(len(op_head)):
        #     for k in range(len(xyz)):
        #         op_joint = op_head[i] + xyz[k]
        #         ma_joint = ma_head[i] + xyz[k]
        #         joint = ma_joint
        #         if xyz[k] == 'Y':
        #             data_vis(op_align_joints, ma_align_joints, joint, op_joint, ma_joint)  # align horizontal(Y) coordinate
        #         elif xyz[k] == 'Z':
        #             data_vis(op_align_joints, ma_align_joints, joint, op_joint, ma_joint)  # align vertical(Z) coordinate
        #         elif xyz[k] == 'X':
        #             data_vis(op_align_joints, ma_align_joints, joint, op_joint, ma_joint)  # align depth(X) coordinate



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
        joint_r.to_csv(rf'/Users/soowan/Documents/PEARL/Data/Data_OB2/SCA_Coordinate/{op_games[game_ind]}/2023{mmdd_p[:4]}-{mmdd_p[-3:]}-{op_games[game_ind]}-Joint_r.csv', encoding = 'utf-8-sig') 
        joint_p.to_csv(rf'/Users/soowan/Documents/PEARL/Data/Data_OB2/SCA_Coordinate/{op_games[game_ind]}/2023{mmdd_p[:4]}-{mmdd_p[-3:]}-{op_games[game_ind]}-Joint_p_val.csv', encoding = 'utf-8-sig') 
        joint_z.to_csv(rf'/Users/soowan/Documents/PEARL/Data/Data_OB2/SCA_Coordinate/{op_games[game_ind]}/2023{mmdd_p[:4]}-{mmdd_p[-3:]}-{op_games[game_ind]}-Joint_z.csv', encoding = 'utf-8-sig') 

        data_r.append(joint_r)
        data_p.append(joint_p)
        data_z.append(joint_z)
        
        print("\nFOLLOWING FILES DO NOT EXIST:", directory_unknown)

        
    # if game doesn't exist for this participant
    try:
      joint_r_overall = pd.concat(data_r)
      joint_p_overall = pd.concat(data_p)
      joint_z_overall = pd.concat(data_z)
    except:
      continue

    # DOWNLOAD the OVERALL joint coordinate correlation R-Z-P-Values --> paste into data results
    joint_r_overall.to_csv(rf'/Users/soowan/Documents/PEARL/Data/Data_OB2/SCA_Coordinate/{op_games[game_ind]}/2023-{op_games[game_ind]}-Joint_r.csv', encoding = 'utf-8-sig')
    joint_p_overall.to_csv(rf'/Users/soowan/Documents/PEARL/Data/Data_OB2/SCA_Coordinate/{op_games[game_ind]}/2023-{op_games[game_ind]}-Joint_p_val.csv', encoding = 'utf-8-sig')
    joint_z_overall.to_csv(rf'/Users/soowan/Documents/PEARL/Data/Data_OB2/SCA_Coordinate/{op_games[game_ind]}/2023-{op_games[game_ind]}-Joint_z.csv', encoding = 'utf-8-sig')
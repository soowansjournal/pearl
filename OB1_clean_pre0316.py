'''
---
# **CHAPTER 3: OB1 AutoClean Pre0316**
---

**Pre 2023-03-16-P05 (_0221-P01, 0314-P02, 0314-P03, 0315-P04_)**  
-  (Y ← OP_X | Z ← OP_Y | X ← OP_Z )

**Post 2023-04-02**
- Check Normality
  - Return **op_highest_peak, ma_highest_peak** from functions **5) Analyze Data**...
  - 1-3) Angle --> min_angle() + max_angle()
  - 1-4) Reach --> peaks_method2() --> reach_lists()
  - 1-5) Speed --> speed_max()

- Store OP Data Tracking Accuracy
  - track_op() --> tracking csv file

**Fixed on 2023-04-11**
- Problem 1: Losing Complete Sight of the User

**5 Bootle Blast + 18 Boot Camp**
- (1-1) Joint Coordinate Position 
- (1-2) Body Segment Length
- (1-3) Joint Angle ROM

**5 Bootle Blast**
- (1-4) Extent of Hand Reach
- (1-5) Max/Mean Hand Speed

Soowan Choi
'''

from OB1_autoclean_pre0316_functions import * # todo import other modules

def load_op(op_file):
  # create dataframe from uploaded csv files using pandas.read_csv()
  op = pd.read_csv(op_file)
  return op 


def load_ma(ma_file):
  # create dataframe from uploaded csv files using pandas.read_csv() & skip the first few rows (3) of information
  ma = pd.read_csv(ma_file, header = 3) 
  return ma

# op_games = ['Power1', 'Power2', 'Wizards', 'War', 'Jet', 'Astro', 'BC1', 'BC2', 'BC3', 'BC4', 'BC5', 'BC6', 'BC7', 'BC8', 'BC9']
# ma_games = ['Power1', 'Power2', 'Wizards', 'War', 'Jet', 'Astro', 'BC1', 'BC2', 'BC3', 'BC4', 'BC5', 'BC6', 'BC7', 'BC8', 'BC9']
# op_games = ['Pediatric', 'Single1', 'Single2', 'Five', 'Thirty']
# ma_games = ['Pediatric', 'Single', 'Single', 'Five', 'Thirty']

op_games = ['Single1', 'Single2', 'Five', 'Thirty']
ma_games = ['Single', 'Single', 'Five', 'Thirty']

# SELECT FILES HERE
# 0221-P01, 0314-P02, 0314-P03, 0315-P04
mmdd = '0314'  
p = 'P02'
mmdd_p = mmdd + '_' + p

for game_ind in range(len(op_games)):
  print(f'\n\n\n\n\n\n\n\n{op_games[game_ind]}\n\n\n\n\n\n\n\n')
  op_file = '2023' + mmdd[:4] + '-' + op_games[game_ind] + "-Data.csv"
  ma_file = '2023' + mmdd[:4] + '-' + ma_games[game_ind] + ".csv"

  # Load OP Data
  op = load_op('/Users/soowan/Documents/PEARL/Data/Data_0551/2023_' + mmdd_p + '/OP_' + mmdd_p + '/' + op_file)
  print(op.head(3))

  # Load MA Data
  ma = load_ma('/Users/soowan/Documents/PEARL/Data/Data_0551/2023_' + mmdd_p + '/MA_' + mmdd_p + '/' + ma_file)
  print(ma.head(3))


  # clean OP
  op_clean, op_hz = clean_op(op)
  op_coord = coord_op(op_clean)
  op_synch = synch_op(op_coord)
  op_track, untracked_op_index, tracking = track_op(op_synch)
  op_filte = filte_op(op_synch, op_hz)


  # clean MA
  ma_clean, ma_hz = clean_ma(ma)
  ma_coord = coord_ma(ma_clean)
  ma_nullv = nullv_ma(ma_coord)
  ma_synch = synch_ma(ma_nullv, op_synch)
  ma_resam = resam_ma(ma_synch, op_synch)
  ma_track = track_ma(ma_resam, untracked_op_index) 
  ma_filte = filte_ma(ma_resam, ma_hz)


  # Problem 1: Losing Complete Sight of the User
  # step 1
  # look at jumps in OP_Synch time stamps
  time_peak_cut = op_synch.Time.iloc[0]
  print(f"Time at 3rd Peak Cut: {round(time_peak_cut,3)}")
  cut_from_time = []
  cut_until_time = []
  for i in range(len(op_synch.Time) - 1):
    time_from = op_synch.Time.iloc[i]
    time_until = op_synch.Time.iloc[i+1]
    time_inc = op_synch.Time.iloc[i+1] - op_synch.Time.iloc[i]
    duration_from = time_from - time_peak_cut
    duration_until = time_until - time_peak_cut
    if time_inc > 1:
      print(f"Index: {i} \tTime from 3rd peak cut: {round(time_from,3)}s \tfor Seconds: {round(time_inc,3)} \tuntil: {round(time_until,3)}s")
      print(f"Index: {i} \tDuration from 3rd peak cut: {round(duration_from,3)}s \tfor Seconds: {round(time_inc,3)} \tuntil duration from 3rd peak cut: {round(duration_until,3)}s\n")
      cut_from_time.append(duration_from)
      cut_until_time.append(duration_until)

  # step 2
  # fix MA if OP seems have lost sight
  if len(cut_until_time) > 0 :

    # step 3
    # locate frames to remove from MA_Synch using OP_Synch jump time increments and MA frequency
    print(f"OP lost sight {len(cut_from_time)} times so cut MA {len(cut_from_time)} times")
    cut_from_frame = []
    cut_until_frame = []
    for i in range(len(cut_from_time)):
      frame_from = ma_hz * cut_from_time[i]  # this many frames from beginning of MA third peak
      frame_until = ma_hz * cut_until_time[i]
      print(f"Cut MA frame from: {round(frame_from,0)} \tuntil frame: {round(frame_until,0)}")
      cut_from_frame.append(frame_from)
      cut_until_frame.append(frame_until)

    # step 4
    # cut MA_Synch data from 3rd peak until specified frames
    print(f"Length of MA_Synch BEFORE removing lost OP sight: {len(ma_synch)}")

    if len(cut_until_frame) == 1:
      ma_synch = ma_synch.drop(np.r_[round(cut_from_frame[0]):round(cut_until_frame[0])])
    elif len(cut_until_frame) == 2:
      ma_synch = ma_synch.drop(np.r_[round(cut_from_frame[0]):round(cut_until_frame[0]), round(cut_from_frame[1]):round(cut_until_frame[1])])
    elif len(cut_until_frame) == 3:
      ma_synch = ma_synch.drop(np.r_[round(cut_from_frame[0]):round(cut_until_frame[0]), round(cut_from_frame[1]):round(cut_until_frame[1]), 
                                    round(cut_from_frame[2]):round(cut_until_frame[2])])

    print(f"Length of MA_Synch AFTER removing lost OP sight: {len(ma_synch)}")

    # step 5
    # resample MA using the newly synchronized MA data
    ma_resam = resam_ma(ma_synch, op_synch)
    ma_track = track_ma(ma_resam, untracked_op_index) 
    ma_filte = filte_ma(ma_resam, ma_hz)

  else:
    print("OP DID NOT LOSE SIGHT OF THE PARTICIPANT")

  
  # Final Data
  op_final = op_filte
  ma_final = ma_filte


  # DOWNLOAD CLEANED OP DATA
  op_final.to_csv(rf'/Users/soowan/Downloads/2023{mmdd}-{p}-{op_games[game_ind]}-Data-OP-CLEAN.csv',  encoding = 'utf-8-sig') 
  # DOWNLOAD OP Data Tracking Accuracy 
  tracking.to_csv(rf'/Users/soowan/Downloads/2023{mmdd}-{p}-{op_games[game_ind]}-Data-tracked.csv', encoding = 'utf-8-sig')
  # DOWNLOAD CLEANED MA BOOT CAMP DATA
  ma_final.to_csv(rf'/Users/soowan/Downloads/2023{mmdd}-{p}-{ma_games[game_ind]}-MA-CLEAN.csv', encoding = 'utf-8-sig') 


  # cut data 
  print(f'OP Synchronized Game Duration: {op_synch.Time.iloc[-1] - op_synch.Time.iloc[0]}')
  print(f'MA Synchronized Game Duration: {ma_synch.Time.iloc[-1] - ma_synch.Time.iloc[0]}')
  print(f'\nOP Frames (NOT Resampled): {op_synch.shape[0]}') 
  print(f'MA Frames (YES Resampled): {ma_resam.shape[0]}') 
  print(f'\nOP Frames (from OP Tracked Data only): {op_track.shape[0]}')  
  print(f'MA Frames (from OP Tracked Data only): {ma_track.shape[0]}') 
  print(f'\nOP Frames (YES Filtered): {op_filte.shape[0]}')  
  print(f'MA Frames (YES Filtered): {ma_filte.shape[0]}\n') 

  op_cut, ma_cut = cut_data(op_final, ma_final)


  # align data using: METHOD 2
  op_align_joints, ma_align_joints = align_joints(op_cut, ma_cut)


  # Visualize ALL Data (39 graphs total)
  op_head = ['Head']
  ma_head = ['Front.Head']
  op_joints = ['Wrist','Hip','Knee']
  ma_joints = ['Wrist','ASIS','Knee']
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

  # Body Data
  for i in range(len(op_joints)):                   # for each joints
    for j in range(len(op_side)):                   # for each sides 
      for k in range(len(xyz)):                     # for each xyz 
        op_joint = op_joints[i] + op_side[j] + xyz[k]  # specific OP joint name
        ma_joint = ma_side[j] + ma_joints[i] + xyz[k]  # specific MA joint name 
        joint = ma_side[j] + ma_joints[i] + xyz[k]     # joint of interest
        if xyz[k] == 'Y':
          data_vis(op_align_joints, ma_align_joints, joint, op_joint, ma_joint)  # align horizontal(Y) coordinate
        elif xyz[k] == 'Z':
          data_vis(op_align_joints, ma_align_joints, joint, op_joint, ma_joint)  # align vertical(Z) coordinate
        elif xyz[k] == 'X':
          data_vis(op_align_joints, ma_align_joints, joint, op_joint, ma_joint)  # align depth(X) coordinate
  
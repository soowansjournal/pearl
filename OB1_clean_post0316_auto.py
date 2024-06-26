'''
---
# **Copy of CHAPTER 3: OB1 Clean Post0316 Auto**
---

**Post 2023-03-16-P05**  
-  (MA_X ← OP_X | MA_Z ← OP_Y | negMA_Y ← OP_Z)
     - 2.2.2) New OP Coordinate   
     - 2.2.4) X-axis is Horizontal (not Y-axis)
     - 3.2.2) Change Direction of the MA Depth Coordinate (Y-axis)
     - 3.2.4) X-axis is Horizontal (not Y-axis)

-  (Y ← OP_X | Z ← OP_Y | X ← OP_Z) + (Y ← MA_X | Z ← MA_Z | X ← negMA_Y)
     - 2.2.2) New OP Coordinate   
     - 3.2.2) New OP Coordinate + Change Direction of the MA Depth Coordinate (X-axis)

**Post 2023-04-02**
- Check Normality
  - Return **op_highest_peak, ma_highest_peak** from functions under **5) Analyze Data**......
  - 1-3) Angle --> min_angle() + max_angle()
  - 1-4) Reach --> peaks_method2() --> reach_lists()
  - 1-5) Speed --> speed_max()

- Store OP Data Tracking Accuracy
  - track_op() --> tracking csv file

**Fixed on 2023-04-11**
- Problem 1: Losing Complete Sight of the User

**Added on 2023-05-22**
- _**Automatically Loop Through Each Participant**_
- For mmdd_p in mmdd_p_all

**Tweaked on 2023-05-22**
- _**Automatically Loop Through Each Game**_
  - _**if directory game file doesn't exist, go to next game**_
  -  except FileNotFoundError: directory_unknown.append()
  - "_**Load automatic peak values to clean**_"
  - def synch_op(op_synch, op_thresh, op_dist, op_peak, op_end)
  - def synch_ma(ma_synch, op_synch, ma_thresh, ma_dist, ma_peak)
  - _**if we dont know the peaks, go to next game**_
  - game_peaks_unknown.append(op_games[game_ind])

**Added on 2023-05-23**
- _**Download Files to Specific Location**_

**5 Bootle Blast + 18 Boot Camp**
- (1-1) Joint Coordinate Position 
- (1-2) Body Segment Length
- (1-3) Joint Angle ROM

**5 Bootle Blast**
- (1-4) Extent of Hand Reach
- (1-5) Max/Mean Hand Speed

Soowan Choi
'''


from OB1_clean_post0316_fun_auto import * # todo import other modules

def load_op(op_file):
  # create dataframe from uploaded csv files using pandas.read_csv()
  op = pd.read_csv(op_file) 

  return op 

def load_ma(ma_file):
  # create dataframe from uploaded csv files using pandas.read_csv() & skip the first few rows (3) of information
  ma = pd.read_csv(ma_file, header = 3) 
  
  return ma





# CREATE FILES (DATE & GAMES)
# ***0404_P10 MA File Named Single1 and Single2 vs Single and Single***
# Files with Problems: 0408_P18_BC8

# op_games = ['Power1', 'Power2', 'Wizards', 'War', 'Jet', 'Astro', 
#             'BC1', 'BC2', 'BC3', 'BC4', 'BC5', 'BC6', 'BC7', 'BC8', 'BC9',
#             'Pediatric', 'Single1', 'Single2', 'Five', 'Thirty']
# ma_games = ['Power1', 'Power2', 'Wizards', 'War', 'Jet', 'Astro', 
#             'BC1', 'BC2', 'BC3', 'BC4', 'BC5', 'BC6', 'BC7', 'BC8', 'BC9',
#             'Pediatric', 'Single', 'Single', 'Five', 'Thirty']
op_games = ['Power1','BC1', 'Single1']
ma_games = ['Power1','BC1', 'Single']

# SELECT FILES HERE
# mmdd_p_all = ['0316_P05', '0322_P06', '0402_P07', '0403_P08', '0403_P09', '0404_P10', '0404_P11', 
#               '0406_P12', '0406_P13', '0407_P14', '0407_P15', '0407_P16', '0408_P17', '0408_P18', 
#               '0411_P19', '0412_P20', '0412_P21', '0413_P22', '0420_P23', '0420_P24', '0430_P25', '0502_P26', '0516_P27', '0601_P28']
mmdd_p_all = ['0601_P28']




directory_unknown = []
game_peaks_unknown = []
# Automatically Loop Through Each Participant
for mmdd_p in mmdd_p_all:
  # Automatically Loop Through Each Game
  for game_ind in range(len(op_games)):
    print(f'\n\n\n\n\n\n\n\n{op_games[game_ind]}\n\n\n\n\n\n\n\n')
    op_file = '2023' + mmdd_p[:4] + '-' + op_games[game_ind] + "-Data.csv"
    ma_file = '2023' + mmdd_p[:4] + '-' + ma_games[game_ind] + ".csv"


    try:
      # Load OP Data
      op = load_op('/Users/soowan/Documents/PEARL/Data/Data_0551/2023_' + mmdd_p + '/OP_' + mmdd_p + '/' + op_file)
      print(op.head(3))

      # Load MA Data
      ma = load_ma('/Users/soowan/Documents/PEARL/Data/Data_0551/2023_' + mmdd_p + '/MA_' + mmdd_p + '/' + ma_file)
      print(ma.head(3))

    except FileNotFoundError:
      # if directory game file doesn't exist, go to next game
      directory_unknown.append('/Users/soowan/Documents/PEARL/Data/Data_0551/2023_' + mmdd_p + '/OP_' + mmdd_p + '/' + op_file)
      continue


    # Load automatic peak values to clean
    peaks = pd.read_csv("/Users/soowan/Documents/VSCODE/Pearl/peaks_post0316.csv") 

    for i in range(len(peaks)):
        if peaks["Date_P##"][i] == mmdd_p:
          if str(peaks[op_games[game_ind]][i]) == 'nan':
              # if we dont know the peaks, go to next game
              game_peaks_unknown.append(mmdd_p + "_" + op_games[game_ind])
              break
          else:
              print(peaks[op_games[game_ind]][i:i+7])
              op_thresh = int(peaks[op_games[game_ind]][i])
              op_dist = int(peaks[op_games[game_ind]][i+1])
              op_peak = int(peaks[op_games[game_ind]][i+2])
              op_end = int(peaks[op_games[game_ind]][i+3])
              ma_thresh = int(peaks[op_games[game_ind]][i+4])
              ma_dist = int(peaks[op_games[game_ind]][i+5])
              ma_peak = int(peaks[op_games[game_ind]][i+6])


              # clean OP
              op_clean, op_hz = clean_op(op)
              op_coord = coord_op(op_clean)
              op_synch = synch_op(op_coord, op_thresh, op_dist, op_peak, op_end)
              op_track, untracked_op_index, tracking = track_op(op_synch)
              op_filte = filte_op(op_synch, op_hz)


              # clean MA
              ma_clean, ma_hz = clean_ma(ma)
              ma_coord = coord_ma(ma_clean)
              ma_nullv = nullv_ma(ma_coord)
              ma_synch = synch_ma(ma_nullv, op_synch, ma_thresh, ma_dist, ma_peak)
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
              op_final = op_synch
              ma_final = ma_resam




              # *** For each participant rename Power1 --> PowerR etc. ***
              if op_games[game_ind] == 'Power1' or op_games[game_ind] == 'Power2':
                  # Load power.csv file to rename
                  power = pd.read_csv("/Users/soowan/Documents/VSCODE/Pearl/power.csv") 

                  # For each row 
                  # If correct participant
                  # For each column
                  # If correct Power column 
                  # Rename: BC#-Game

                  for i in range(len(power)):
                      if mmdd_p in power.iloc[i,0]:
                          for j in range(len(power.columns)):
                              if op_games[game_ind] == str(power.columns[j]):
                                  op_game = str(power.iloc[i,j])
                                  ma_game = str(power.iloc[i,j])
                                  break
                              


              # *** For each participant rename BC#-Game ***
              if 'BC' in op_games[game_ind]:
                # Load bootcamp.csv file to rename
                bootcamp = pd.read_csv("/Users/soowan/Documents/VSCODE/Pearl/bootcamp.csv") 

                # For each row 
                # If correct participant
                # For each column
                # If correct BC column and corresponding cell isn't empty
                # Rename: BC#-Game
                for i in range(len(bootcamp)):
                    if mmdd_p in bootcamp.iloc[i,0]:
                        for j in range(len(bootcamp.columns)):
                            if op_games[game_ind] == str(bootcamp.columns[j]):
                                    if str(bootcamp.iloc[i,j]) != 'nan':
                                        op_game = op_games[game_ind] + '-' + str(bootcamp.iloc[i,j])
                                        ma_game = ma_games[game_ind] + '-' + str(bootcamp.iloc[i,j])
                                        break
                                    else:
                                        op_game = op_games[game_ind] + '-' + 'NA'
                                        ma_game = ma_games[game_ind] + '-' + 'NA'
                                        break


              # DOWNLOAD FILES TO DOWNLOADS FOLDER
              if 'Power' in op_games[game_ind] or 'BC' in op_games[game_ind]:
                # DOWNLOAD CLEANED OP DATA
                op_final.to_csv(rf'/Users/soowan/Downloads/2023{mmdd_p[:4]}-{mmdd_p[-3:]}-{op_game}-Data-OP-CLEAN.csv',  encoding = 'utf-8-sig') 
                # DOWNLOAD OP Data Tracking Accuracy 
                tracking.to_csv(rf'/Users/soowan/Downloads/2023{mmdd_p[:4]}-{mmdd_p[-3:]}-{op_game}-Data-tracked.csv', encoding = 'utf-8-sig')
                # DOWNLOAD CLEANED MA BOOT CAMP DATA
                ma_final.to_csv(rf'/Users/soowan/Downloads/2023{mmdd_p[:4]}-{mmdd_p[-3:]}-{ma_game}-MA-CLEAN.csv', encoding = 'utf-8-sig')
              else:
                op_final.to_csv(rf'/Users/soowan/Downloads/2023{mmdd_p[:4]}-{mmdd_p[-3:]}-{op_games[game_ind]}-Data-OP-CLEAN.csv',  encoding = 'utf-8-sig') 
                # DOWNLOAD OP Data Tracking Accuracy 
                tracking.to_csv(rf'/Users/soowan/Downloads/2023{mmdd_p[:4]}-{mmdd_p[-3:]}-{op_games[game_ind]}-Data-tracked.csv', encoding = 'utf-8-sig')
                # DOWNLOAD CLEANED MA BOOT CAMP DATA
                ma_final.to_csv(rf'/Users/soowan/Downloads/2023{mmdd_p[:4]}-{mmdd_p[-3:]}-{op_games[game_ind]}-MA-CLEAN.csv', encoding = 'utf-8-sig')
                 

              # # DOWNLOAD FILES TO SPECIFIC LOCATION
              # if 'Power' in op_games[game_ind] or 'BC' in op_games[game_ind]:
              #   # DOWNLOAD CLEANED OP DATA
              #   op_final.to_csv(rf'/Users/soowan/Documents/PEARL/Data/Data_0551/2023_{mmdd_p[:4]}_{mmdd_p[-3:]}/Auto_Clean_{mmdd_p[:4]}_{mmdd_p[-3:]}/2023{mmdd_p[:4]}-{mmdd_p[-3:]}-{op_game}-Data-OP-CLEAN.csv',  encoding = 'utf-8-sig') 
              #   # DOWNLOAD OP Data Tracking Accuracy 
              #   tracking.to_csv(rf'/Users/soowan/Documents/PEARL/Data/Data_0551/2023_{mmdd_p[:4]}_{mmdd_p[-3:]}/Auto_Results_{mmdd_p[:4]}_{mmdd_p[-3:]}/2023{mmdd_p[:4]}-{mmdd_p[-3:]}-{op_game}-Data-tracked.csv', encoding = 'utf-8-sig')
              #   # DOWNLOAD CLEANED MA BOOT CAMP DATA
              #   ma_final.to_csv(rf'/Users/soowan/Documents/PEARL/Data/Data_0551/2023_{mmdd_p[:4]}_{mmdd_p[-3:]}/Auto_Clean_{mmdd_p[:4]}_{mmdd_p[-3:]}/2023{mmdd_p[:4]}-{mmdd_p[-3:]}-{ma_game}-MA-CLEAN.csv', encoding = 'utf-8-sig') 
              # else:
              #   # DOWNLOAD CLEANED OP DATA
              #   op_final.to_csv(rf'/Users/soowan/Documents/PEARL/Data/Data_0551/2023_{mmdd_p[:4]}_{mmdd_p[-3:]}/Auto_Clean_{mmdd_p[:4]}_{mmdd_p[-3:]}/2023{mmdd_p[:4]}-{mmdd_p[-3:]}-{op_games[game_ind]}-Data-OP-CLEAN.csv',  encoding = 'utf-8-sig') 
              #   # DOWNLOAD OP Data Tracking Accuracy 
              #   tracking.to_csv(rf'/Users/soowan/Documents/PEARL/Data/Data_0551/2023_{mmdd_p[:4]}_{mmdd_p[-3:]}/Auto_Results_{mmdd_p[:4]}_{mmdd_p[-3:]}/2023{mmdd_p[:4]}-{mmdd_p[-3:]}-{op_games[game_ind]}-Data-tracked.csv', encoding = 'utf-8-sig')
              #   # DOWNLOAD CLEANED MA BOOT CAMP DATA
              #   ma_final.to_csv(rf'/Users/soowan/Documents/PEARL/Data/Data_0551/2023_{mmdd_p[:4]}_{mmdd_p[-3:]}/Auto_Clean_{mmdd_p[:4]}_{mmdd_p[-3:]}/2023{mmdd_p[:4]}-{mmdd_p[-3:]}-{op_games[game_ind]}-MA-CLEAN.csv', encoding = 'utf-8-sig') 
                 


              # cut data 
              # print(f'OP Synchronized Game Duration: {op_synch.Time.iloc[-1] - op_synch.Time.iloc[0]}')
              # print(f'MA Synchronized Game Duration: {ma_synch.Time.iloc[-1] - ma_synch.Time.iloc[0]}')
              # print(f'\nOP Frames (NOT Resampled): {op_synch.shape[0]}') 
              # print(f'MA Frames (YES Resampled): {ma_resam.shape[0]}') 
              # print(f'\nOP Frames (from OP Tracked Data only): {op_track.shape[0]}')  
              # print(f'MA Frames (from OP Tracked Data only): {ma_track.shape[0]}') 
              # print(f'\nOP Frames (YES Filtered): {op_filte.shape[0]}')  
              # print(f'MA Frames (YES Filtered): {ma_filte.shape[0]}\n') 

              op_cut, ma_cut = cut_data(op_final, ma_final)


              # align data using: METHOD 2
              op_align_joints, ma_align_joints = align_joints(op_cut, ma_cut)


              # Visualize ALL Data (39 graphs total)
              op_head = ['Head']
              ma_head = ['Front.Head']
              op_joints = ['Wrist', 'Hip', 'Knee']
              ma_joints = ['Wrist', 'ASIS', 'Knee']
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


print("\nFOLLOWING FILES DO NOT EXIST:", directory_unknown)
# for dir in directory_unknown:
#   print(dir)

print("\nFOLLOWING GAMES HAVE UNKNOWN PEAKS:", game_peaks_unknown)
# for games in game_peaks_unknown:
#   print(games)

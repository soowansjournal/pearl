'''
---
# **OB1 AutoAnalysis 1-4) Extent of Hand Reach**
---

### **_Use for all ANALYSIS since cleaned data is all SAME FORMAT_**

Soowan Choi
'''


from OB1_autoanalysis_4_functions import * # todo import other modules


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
mmdd = '0314' 
p = 'P02'
mmdd_p = mmdd + '_' + p


directory_unknown = []
data = []

for game_ind in range(len(op_games)):
    print(f'\n\n\n\n\n\n\n\n{op_games[game_ind]}\n\n\n\n\n\n\n\n')
    op_file = '2023' + mmdd + '-' + p + '-' + op_games[game_ind] + "-Data-OP-CLEAN.csv"
    ma_file = '2023' + mmdd + '-' + p + '-' + ma_games[game_ind] + "-MA-CLEAN.csv"
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


    # 1-4) Extent of Hand Reach

    op_array_left, ma_array_left, op_array_right, ma_array_right = reach_calculations(op_final, ma_final)

    # left reach lists (remove outliers)
    side = 'Left'
    op_left_max, ma_left_max, diff_left_max, per_left_max, op_highest_leftreach, ma_highest_leftreach = reach_lists(side, op_array_left, ma_array_left)

    # right reach lists (remove outliers)
    side = 'Right'
    op_right_max, ma_right_max, diff_right_max, per_right_max, op_highest_rightreach, ma_highest_rightreach = reach_lists(side, op_array_right, ma_array_right)

    # show results
    order = ['X', 'Y', 'Z', '3D']

    left_reach_max = table_left_max(order, op_left_max, ma_left_max, diff_left_max, per_left_max)
    right_reach_max = table_right_max(order, op_right_max, ma_right_max, diff_right_max, per_right_max)

    reach = table_reach_results(2, left_reach_max, right_reach_max)

    # # DOWNLOAD the reach results --> paste into data results
    # reach.to_csv(rf'/Users/soowan/Downloads/2023{mmdd}-{p}-{op_games[game_ind]}-reach.csv', encoding = 'utf-8-sig') 

    
    # Reorganize DataFrame Results
    dataframes = []
    side_coord = ['L.Max.X', 'L.Max.Y', 'L.Max.Z', 'L.Max.3D', 'R.Max.X', 'R.Max.Y', 'R.Max.Z', 'R.Max.3D']
    specify = ['(OP)', '(MA)', '(Diff)', '(Error%)']
    new_col = []

    # Create New Column Names
    for i in side_coord:
        for j in specify:
            name = i + j
            new_col.append(name)

    # Recreate into single row
    for i in range(len(left_reach_max)):
        left = pd.DataFrame(np.array(left_reach_max.iloc[i,1:]))
        dataframes.append(left.transpose())
        
    for i in range(len(right_reach_max)):
        right = pd.DataFrame(np.array(left_reach_max.iloc[i,1:]))
        dataframes.append(right.transpose())

    reach_re = pd.concat(dataframes, axis = 1)
    reach_re = np.array(reach_re[0:])
    reach_re = pd.DataFrame(reach_re, columns = new_col, index = [mmdd_p[-3:]])


    # DOWNLOAD the reach results --> paste into data results
    reach_re.to_csv(rf'/Users/soowan/Downloads/2023{mmdd}-{p}-{op_games[game_ind]}-reach.csv', encoding = 'utf-8-sig')

    data.append(reach_re)









    def check_normality(data, title):
        plt.hist(data, bins= 30, density=True, alpha= 0.5, color='b')
        plt.title(title)
        plt.xlabel("Reach [cm]")
        plt.ylabel("Frequency")
        # Fit a normal distribution to the data (mean and standard deviation)
        mu, std = norm.fit(data)
        # Plot the PDF.
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        plt.plot(x, p, 'k', linewidth=2)


    # Check Normality 1-4)

    index = ['# Samples', 'SW (Statistic)', 'SW (p-val)', 'KS (Statistic)', 'KS (p-val)', 'Mean', 'Median']
    cols = ['OP Left Reach', 'MA Left Reach', 'OP Right Reach', 'MA Right Reach']

    num_sam = [len(op_highest_leftreach), len(ma_highest_leftreach),
                len(op_highest_rightreach), len(ma_highest_rightreach)]

    # Shapiro Wilk Test of Distribution
    sw_stat = [round(shapiro(op_highest_leftreach)[0],3), round(shapiro(ma_highest_leftreach)[0],3),
                round(shapiro(op_highest_rightreach)[0],3), round(shapiro(ma_highest_rightreach)[0],3)]

    sw_pval = [round(shapiro(op_highest_leftreach)[1],3), round(shapiro(ma_highest_leftreach)[1],3),
                round(shapiro(op_highest_rightreach)[1],3), round(shapiro(ma_highest_rightreach)[1],3)]

    # Kolmogorov-Smirnov Test of Distribution
    ks_stat = [round(kstest(op_highest_leftreach,'norm')[0],3), round(kstest(ma_highest_leftreach,'norm')[0],3),
                round(kstest(op_highest_rightreach,'norm')[0],3), round(kstest(ma_highest_rightreach,'norm')[0],3)]

    ks_pval = [round(kstest(op_highest_leftreach,'norm')[1],3), round(kstest(ma_highest_leftreach,'norm')[1],3),
                round(kstest(op_highest_rightreach,'norm')[1],3), round(kstest(ma_highest_rightreach,'norm')[1],3)]

    # Mean and Median of Distribution
    means = [round(op_highest_leftreach.mean(),3), round(ma_highest_leftreach.mean(),3),
                round(op_highest_rightreach.mean(),3), round(ma_highest_rightreach.mean(),3)]

    medians = [round(np.median(op_highest_leftreach),3), round(np.median(ma_highest_leftreach),3),
                round(np.median(op_highest_rightreach),3), round(np.median(ma_highest_rightreach),3)]

    pd.DataFrame([num_sam, sw_stat, sw_pval, ks_stat, ks_pval, means, medians], index = index, columns = cols)


    # Check Normality 1-4)

    plt.rcParams["figure.figsize"] = [12,8]

    plt.subplot(2,2,1)
    check_normality(op_highest_leftreach,"OP Left Reach")
    plt.subplot(2,2,2)
    check_normality(ma_highest_leftreach,"MA Left Reach")
    plt.subplot(2,2,3)
    check_normality(op_highest_rightreach,"OP Right Reach")
    plt.subplot(2,2,4)
    check_normality(ma_highest_rightreach,"MA Right Reach")

    # using padding
    plt.tight_layout(pad=3.0)

    plt.show()

    # pd.DataFrame(op_highest_peak_left).to_csv(f'tmp_normality.csv', encoding = 'utf-8-sig') 
    # files.download(f'tmp_normality.csv')


    # Check Normality 1-4)
    plt.rcParams["figure.figsize"] = [15,8]

    plt.subplot(2,4,1)
    probplot(op_highest_leftreach, dist="norm", plot = pylab)
    plt.title("OP Left Reach")
    plt.subplot(2,4,2)
    probplot(ma_highest_leftreach, dist="norm", plot = pylab)
    plt.title("MA Left Reach")
    plt.subplot(2,4,3)
    probplot(op_highest_rightreach, dist="norm", plot = pylab)
    plt.title("OP Right Reach")
    plt.subplot(2,4,4)
    probplot(ma_highest_rightreach, dist="norm", plot = pylab)
    plt.title("MA Right Reach")

    # using padding
    plt.tight_layout(pad=3.0)

    plt.show()





print("\nFOLLOWING FILES DO NOT EXIST:", directory_unknown)
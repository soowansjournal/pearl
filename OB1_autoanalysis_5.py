'''
---
# **OB1 AutoAnalysis 1-5) Maximum/Mean Hand Speed**
---

### **_Use for all ANALYSIS since cleaned data is all SAME FORMAT_**

Soowan Choi
'''


from OB1_autoanalysis_5_functions import * # todo import other modules


def load_op(op_file):
  # create dataframe from uploaded csv files using pandas.read_csv()
  op = pd.read_csv(op_file) 
  return op 


def load_ma(ma_file):
  # create dataframe from uploaded csv files using pandas.read_csv() & skip the first few rows (3) of information
  ma = pd.read_csv(ma_file) 
  return ma


op_games = ['PowerR', 'PowerL', 'Wizards', 'War', 'Jet', 'Astro']

ma_games = ['PowerR', 'PowerL', 'Wizards', 'War', 'Jet', 'Astro']


# SELECT FILES HERE
mmdd = '0314' 
p = 'P02'
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


    # 1-5) Maximum/Mean Hand Speed

    # speed arrays
    op_all_left, ma_all_left, op_all_right, ma_all_right = speed_calculations(op_final, ma_final)


    # remove outliers
    op_all_left = remove_outliers(pd.Series(op_all_left))
    ma_all_left = remove_outliers(pd.Series(ma_all_left))
    op_all_right = remove_outliers(pd.Series(op_all_right))
    ma_all_right = remove_outliers(pd.Series(ma_all_right))


    # left vel MAX
    op_leftvel_max, ma_leftvel_max, diff_leftvel_max, per_leftvel_max, op_highest_leftvel, ma_highest_leftvel = speed_max(op_all_left, ma_all_left, 'LEFT')
    # left vel MEAN
    op_leftvel_mean, ma_leftvel_mean, diff_leftvel_mean, per_leftvel_mean = speed_mean(op_all_left, ma_all_left)

    # right vel MAX
    op_rightvel_max, ma_rightvel_max, diff_rightvel_max, per_rightvel_max, op_highest_rightvel, ma_highest_rightvel = speed_max(op_all_right, ma_all_right, 'RIGHT')
    # right vel MEAN
    op_rightvel_mean, ma_rightvel_mean, diff_rightvel_mean, per_rightvel_mean = speed_mean(op_all_right, ma_all_right)


    # show results
    left_vel_max = table_vel(op_leftvel_max, ma_leftvel_max, diff_leftvel_max, per_leftvel_max, 'L', 'MAX', order = 'NA')
    left_vel_mean = table_vel(op_leftvel_mean, ma_leftvel_mean, diff_leftvel_mean, per_leftvel_mean, 'L', 'MEAN', order = 'NA')

    right_vel_max = table_vel(op_rightvel_max, ma_rightvel_max, diff_rightvel_max, per_rightvel_max, 'R', 'MAX', order = 'NA')
    right_vel_mean = table_vel(op_rightvel_mean, ma_rightvel_mean, diff_rightvel_mean, per_rightvel_mean, 'R', 'MEAN', order = 'NA')

    speed = table_vel_results('1', left_vel_max, left_vel_mean, right_vel_max, right_vel_mean)

    # DOWNLOAD the speed results --> paste into data results
    speed.to_csv(rf'/Users/soowan/Downloads/2023{mmdd}-{p}-{op_games[game_ind]}-speed.csv', encoding = 'utf-8-sig') 


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


    # Check Normality 1-5)

    index = ['# Samples', 'SW (Statistic)', 'SW (p-val)', 'KS (Statistic)', 'KS (p-val)', 'Mean', 'Median']
    cols = ['OP Left MaxVel', 'MA Left MaxVel', 'OP Left MeanVel', 'MA Left MeanVel', 'OP Right MaxVel', 'MA Right MaxVel', 'OP Right MeanVel', 'MA Right MeanVel']

    # Number of Samples
    num_sam = [len(op_highest_leftvel), len(ma_highest_leftvel),
                len(op_highest_rightvel), len(ma_highest_rightvel),
            len(op_all_left), len(ma_all_left),
                len(op_all_right), len(ma_all_right)]

    # Shapiro Wilk Test of Distribution
    sw_stat = [round(shapiro(op_highest_leftvel)[0],3), round(shapiro(ma_highest_leftvel)[0],3),
                round(shapiro(op_highest_rightvel)[0],3), round(shapiro(ma_highest_rightvel)[0],3),
                round(shapiro(op_all_left)[0],3), round(shapiro(ma_all_left)[0],3),
                round(shapiro(op_all_right)[0],3), round(shapiro(ma_all_right)[0],3)]

    sw_pval = [round(shapiro(op_highest_leftvel)[1],3), round(shapiro(ma_highest_leftvel)[1],3),
                round(shapiro(op_highest_rightvel)[1],3), round(shapiro(ma_highest_rightvel)[1],3),
                round(shapiro(op_all_left)[1],3), round(shapiro(ma_all_left)[1],3),
                round(shapiro(op_all_right)[1],3), round(shapiro(ma_all_right)[1],3)]

    # Kolmogorov-Smirnov Test of Distribution
    ks_stat = [round(kstest(op_highest_leftvel,'norm')[0],3), round(kstest(ma_highest_leftvel,'norm')[0],3),
                round(kstest(op_highest_rightvel,'norm')[0],3), round(kstest(ma_highest_rightvel,'norm')[0],3),
                round(kstest(op_all_left,'norm')[0],3), round(kstest(ma_all_left,'norm')[0],3),
                round(kstest(op_all_right,'norm')[0],3), round(kstest(ma_all_right,'norm')[0],3)]

    ks_pval = [round(kstest(op_highest_leftvel,'norm')[1],3), round(kstest(ma_highest_leftvel,'norm')[1],3),
                round(kstest(op_highest_rightvel,'norm')[1],3), round(kstest(ma_highest_rightvel,'norm')[1],3),
                round(kstest(op_all_left,'norm')[1],3), round(kstest(ma_all_left,'norm')[1],3),
                round(kstest(op_all_right,'norm')[1],3), round(kstest(ma_all_right,'norm')[1],3)]

    # Mean and Median of Distribution
    means = [round(op_highest_leftvel.mean(),3), round(ma_highest_leftvel.mean(),3),
                round(op_highest_rightvel.mean(),3), round(ma_highest_rightvel.mean(),3),
            round(op_all_left.mean(),3), round(ma_all_left.mean(),3),
                round(op_all_right.mean(),3), round(ma_all_right.mean(),3)]

    medians = [round(np.median(op_highest_leftvel),3), round(np.median(ma_highest_leftvel),3),
                round(np.median(op_highest_rightvel),3), round(np.median(ma_highest_rightvel),3),
            round(np.median(op_all_left),3), round(np.median(ma_all_left),3),
                round(np.median(op_all_right),3), round(np.median(ma_all_right),3)]

    pd.DataFrame([num_sam, sw_stat, sw_pval, ks_stat, ks_pval, means, medians], index = index, columns = cols)


    # Check Normality 1-5)

    plt.rcParams["figure.figsize"] = [15,8]

    plt.subplot(2,4,1)
    check_normality(op_highest_leftvel,"OP Left MaxVel")
    plt.subplot(2,4,2)
    check_normality(ma_highest_leftvel,"MA Left MaxVel")
    plt.subplot(2,4,3)
    check_normality(op_highest_rightvel,"OP Right MaxVel")
    plt.subplot(2,4,4)
    check_normality(ma_highest_rightvel,"MA Right MaxVel")
    plt.subplot(2,4,5)
    check_normality(op_all_left,"OP Left MeanVel")
    plt.subplot(2,4,6)
    check_normality(ma_all_left,"MA Left MeanVel")
    plt.subplot(2,4,7)
    check_normality(op_all_right,"OP Right MeanVel")
    plt.subplot(2,4,8)
    check_normality(ma_all_right,"MA Right MeanVel")

    # using padding
    plt.tight_layout(pad=3.0)

    plt.show()


    # pd.DataFrame(op_highest_peak_left).to_csv(f'tmp_normality.csv', encoding = 'utf-8-sig') 
    # files.download(f'tmp_normality.csv')


    # Check Normality 1-5)
    plt.rcParams["figure.figsize"] = [15,8]

    plt.subplot(2,4,1)
    probplot(op_highest_leftvel, dist="norm", plot = pylab)
    plt.title("OP Left MaxVel")
    plt.subplot(2,4,2)
    probplot(ma_highest_leftvel, dist="norm", plot = pylab)
    plt.title("MA Left MaxVel")
    plt.subplot(2,4,3)
    probplot(op_highest_rightvel, dist="norm", plot = pylab)
    plt.title("OP Right MaxVel")
    plt.subplot(2,4,4)
    probplot(ma_highest_rightvel, dist="norm", plot = pylab)
    plt.title("MA Right MaxVel")
    plt.subplot(2,4,5)
    probplot(op_all_left, dist="norm", plot = pylab)
    plt.title("OP Left MeanVel")
    plt.subplot(2,4,6)
    probplot(ma_all_left, dist="norm", plot = pylab)
    plt.title("MA Left MeanVel")
    plt.subplot(2,4,7)
    probplot(op_all_right, dist="norm", plot = pylab)
    plt.title("OP Right MeanVel")
    plt.subplot(2,4,8)
    probplot(ma_all_right, dist="norm", plot = pylab)
    plt.title("MA Right MeanVel")

    # using padding
    plt.tight_layout(pad=3.0)

    plt.show()
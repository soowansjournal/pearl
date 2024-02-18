# peaks_method2() --> Avg of Top 5% Peak Values

        tmp = []
        op_peaks = []
        # absolute max OP peak
        op_single_peak = op_array_test.max()
        # top 5% are the highest peak values
        for ind,val in enumerate(op_array_test):
            if val/op_single_peak > 0.95:
                tmp.append(val)
                op_peaks.append(ind)
        op_highest_peak = np.array(tmp)
        # average max OP peak
        op_average_peak = round(op_highest_peak.mean(),3)
        print(f"Absolute Max OP Peak:\t {op_single_peak} \nAverage Max OP Peak:\t {op_average_peak} cm")

        tmp = []
        ma_peaks = []
        # absolute max MA peak
        ma_single_peak = ma_array_test.max()
        # top 5% are the highest peak values
        for ind,val in enumerate(ma_array_test):
            if val/ma_single_peak > 0.95:
                tmp.append(val)
                ma_peaks.append(ind)
        ma_highest_peak = np.array(tmp)
        # average max MA peak
        ma_average_peak = round(ma_highest_peak.mean(),3)
        print(f"Absolute Max MA Peak:\t {ma_single_peak} \nAverage Max MA Peak:\t {ma_average_peak} cm")
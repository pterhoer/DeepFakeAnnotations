import pandas as pd
from sklearn.metrics import roc_curve
import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib
import sys
import os


all_attributes = 'A-DFDC.csv'   # Available in the google drive
test_output = 'inputs/dfdc_capnet/dfdc_capnet.txt'  # Available in the google drive, test folder
dataset = 'DFDC_capnet'
output_file = 'outputs/DFDC_capnet.csv'


def main():

    # DFDC DFDC_train_single / DFDC_validation_single -> DFDC_validation_single
    all_data = pd.read_csv(all_attributes, index_col=0)
    all_data['path'] = all_data['path'].str.replace('DFDC_train_single/DFDC_validation_single', 'DFDC_validation_single')
    test_data = pd.read_csv(test_output, delimiter="\t", index_col=False)

    data = test_data.merge(all_data, how='left', left_on='image_path', right_on='path')
    data.drop(columns=['path', 'path_folder', 'label'], inplace=True)
    df_real = data.loc[data['real_label'] == 0]
    df_fake = data.loc[data['fake_label'] == 1]

    columns_name = ["male", "young", "asian", "white", "black", "shiny_skin", "bald", "wavy_hair", "receding_hairline",
                    "bangs", "black_hair", "blond_hair", "no_beard", "mustache", "goatee", "oval_face", "square_face",
                    "double_chain", "chubby", "obstructed_forehead", "fully_visible_forehead", "mouth_closed",
                    "smiling", "big_lips", "big_nose", "pointy_nose", "heavy_makeup", "wearing_hat",
                    "wearing_lipstick", "eyeglasses", "attractive"]

    matrix = ['TPR@FPR-10%', 'TPR@FPR-1%', 'TPR@FPR-0.1%', 'Error rate', 'Balanced Error rate', 'F1-score', 'Err (Real)',
              'Err (Fake)']
    group = ['Data', 'Control']
    columns = pd.MultiIndex.from_product([matrix, group])
    # columns = columns.append(['real_acc', 'fake_acc'])
    attributes = columns_name
    positive = ['Positive', 'Negative', 'Rel. Perf.']
    indexes = pd.MultiIndex.from_product([attributes, positive])

    df_results = pd.DataFrame(columns=columns, index=indexes)

    for column_name in columns_name:
        print(column_name)
        df1 = data.loc[data[column_name] == 1, ['real_label', 'prediction_score', 'output_label']]
        df2 = data.loc[data[column_name] == -1, ['real_label', 'prediction_score', 'output_label']]
        df_real_positive = df_real.loc[data[column_name] == 1, ['real_label', 'output_label']]
        df_real_negative = df_real.loc[data[column_name] == -1, ['real_label', 'output_label']]
        df_fake_positive = df_fake.loc[data[column_name] == 1, ['real_label', 'output_label']]
        df_fake_negative = df_fake.loc[data[column_name] == -1, ['real_label', 'output_label']]

        df1_control = data[['real_label', 'prediction_score', 'output_label']].sample(n=len(df1), random_state=15)
        df2_control = data[['real_label', 'prediction_score', 'output_label']].sample(n=len(df2), random_state=15)

        df_real_positive_control = df_real[['real_label', 'prediction_score', 'output_label']].sample(n=len(df_real_positive), random_state=15)
        df_real_negative_control = df_real[['real_label', 'prediction_score', 'output_label']].sample(n=len(df_real_negative), random_state=15)
        df_fake_positive_control = df_fake[['real_label', 'prediction_score', 'output_label']].sample(n=len(df_fake_positive), random_state=15)
        df_fake_negative_control = df_fake[['real_label', 'prediction_score', 'output_label']].sample(n=len(df_fake_negative), random_state=15)

        # if len(df1) > 100 and len(df2) > 100:
        if len(df1['real_label'].unique()) == 2 and len(df2['real_label'].unique()) == 2:
            print(len(df1['real_label'].unique()), len(df2['real_label'].unique()))

            tpr_01_p, tpr_001_p, tpr_0001_p = cal_frr_at_far(df1['real_label'], df1['prediction_score'], column_name, 1)
            tpr_01_n, tpr_001_n, tpr_0001_n = cal_frr_at_far(df2['real_label'], df2['prediction_score'], column_name, 2)
            tpr_01_cp, tpr_001_cp, tpr_0001_cp = cal_frr_at_far(df1_control['real_label'], df1_control['prediction_score'], column_name, 3)
            tpr_01_cn, tpr_001_cn, tpr_0001_cn = cal_frr_at_far(df2_control['real_label'], df2_control['prediction_score'], column_name, 4)

            acc_p, balanced_acc_p, f1_score_p = cal_acc(df1['output_label'], df1['real_label'])
            acc_n, balanced_acc_n, f1_score_n = cal_acc(df2['output_label'], df2['real_label'])
            acc_cp, balanced_acc_cp, f1_score_cp = cal_acc(df1_control['output_label'], df1_control['real_label'])
            acc_cn, balanced_acc_cn, f1_score_cn = cal_acc(df2_control['output_label'], df2_control['real_label'])

            df_results.loc[column_name, 'Positive'].loc['TPR@FPR-10%', 'Data'] = tpr_01_p
            df_results.loc[column_name, 'Negative'].loc['TPR@FPR-10%', 'Data'] = tpr_01_n
            df_results.loc[column_name, 'Rel. Perf.'].loc['TPR@FPR-10%', 'Data'] = 1 - tpr_01_p/tpr_01_n
            df_results.loc[column_name, 'Positive'].loc['TPR@FPR-1%', 'Data'] = tpr_001_p
            df_results.loc[column_name, 'Negative'].loc['TPR@FPR-1%', 'Data'] = tpr_001_n
            df_results.loc[column_name, 'Rel. Perf.'].loc['TPR@FPR-1%', 'Data'] = 1 - tpr_001_p / tpr_001_n
            df_results.loc[column_name, 'Positive'].loc['TPR@FPR-0.1%', 'Data'] = tpr_0001_p
            df_results.loc[column_name, 'Negative'].loc['TPR@FPR-0.1%', 'Data'] = tpr_0001_n
            df_results.loc[column_name, 'Rel. Perf.'].loc['TPR@FPR-0.1%', 'Data'] = 1 - tpr_0001_p / tpr_0001_n
            df_results.loc[column_name, 'Positive'].loc['Error rate', 'Data'] = 1 - acc_p
            df_results.loc[column_name, 'Negative'].loc['Error rate', 'Data'] = 1 - acc_n
            if acc_n != 1:
                df_results.loc[column_name, 'Rel. Perf.'].loc['Error rate', 'Data'] = 1 - (1 - acc_p) / (1 - acc_n)
            else:
                df_results.loc[column_name, 'Rel. Perf.'].loc['Error rate', 'Data'] = 'All correct'
            df_results.loc[column_name, 'Positive'].loc['Balanced Error rate', 'Data'] = 1 - balanced_acc_p
            df_results.loc[column_name, 'Negative'].loc['Balanced Error rate', 'Data'] = 1 - balanced_acc_n
            if balanced_acc_n != 1:
                df_results.loc[column_name, 'Rel. Perf.'].loc['Balanced Error rate', 'Data'] = 1 - (1 - balanced_acc_p) / (1 - balanced_acc_n)
            else:
                df_results.loc[column_name, 'Rel. Perf.'].loc['Balanced Error rate', 'Data'] = 'All correct'

            if isinstance(f1_score_p, float) and isinstance(f1_score_n, float):
                df_results.loc[column_name, 'Positive'].loc['F1-score', 'Data'] = f1_score_p
                df_results.loc[column_name, 'Negative'].loc['F1-score', 'Data'] = f1_score_n
                df_results.loc[column_name, 'Rel. Perf.'].loc['F1-score', 'Data'] = 1 - f1_score_p / f1_score_n
            else:
                df_results.loc[column_name, 'Positive'].loc['F1-score', 'Data'] = 'no tp'
                df_results.loc[column_name, 'Negative'].loc['F1-score', 'Data'] = 'no tp'
                df_results.loc[column_name, 'Rel. Perf.'].loc['F1-score', 'Data'] = 'no tp'

            df_results.loc[column_name, 'Positive'].loc['TPR@FPR-10%', 'Control'] = tpr_01_cp
            df_results.loc[column_name, 'Negative'].loc['TPR@FPR-10%', 'Control'] = tpr_01_cn
            df_results.loc[column_name, 'Rel. Perf.'].loc['TPR@FPR-10%', 'Control'] = 1 - tpr_01_cp / tpr_01_cn
            df_results.loc[column_name, 'Positive'].loc['TPR@FPR-1%', 'Control'] = tpr_001_cp
            df_results.loc[column_name, 'Negative'].loc['TPR@FPR-1%', 'Control'] = tpr_001_cn
            df_results.loc[column_name, 'Rel. Perf.'].loc['TPR@FPR-1%', 'Control'] = 1 - tpr_001_cp / tpr_001_cn
            df_results.loc[column_name, 'Positive'].loc['TPR@FPR-0.1%', 'Control'] = tpr_0001_cp
            df_results.loc[column_name, 'Negative'].loc['TPR@FPR-0.1%', 'Control'] = tpr_0001_cn
            df_results.loc[column_name, 'Rel. Perf.'].loc['TPR@FPR-0.1%', 'Control'] = 1 - tpr_0001_cp / tpr_0001_cn
            df_results.loc[column_name, 'Positive'].loc['Error rate', 'Control'] = 1 - acc_cp
            df_results.loc[column_name, 'Negative'].loc['Error rate', 'Control'] = 1 - acc_cn
            if acc_cn != 1:
                df_results.loc[column_name, 'Rel. Perf.'].loc['Error rate', 'Control'] = 1 - (1 - acc_cp) / (1 - acc_cn)
            else:
                df_results.loc[column_name, 'Rel. Perf.'].loc['Error rate', 'Control'] = 'All correct'
            df_results.loc[column_name, 'Positive'].loc['Balanced Error rate', 'Control'] = 1 - balanced_acc_cp
            df_results.loc[column_name, 'Negative'].loc['Balanced Error rate', 'Control'] = 1 - balanced_acc_cn
            if balanced_acc_cn != 1:
                df_results.loc[column_name, 'Rel. Perf.'].loc['Balanced Error rate', 'Control'] = 1 - (1 - balanced_acc_cp) / (1 - balanced_acc_cn)
            else:
                df_results.loc[column_name, 'Rel. Perf.'].loc['Balanced Error rate', 'Control'] = 'All correct'

            if isinstance(f1_score_cp, float) and isinstance(f1_score_cn, float):
                df_results.loc[column_name, 'Positive'].loc['F1-score', 'Control'] = f1_score_cp
                df_results.loc[column_name, 'Negative'].loc['F1-score', 'Control'] = f1_score_cn
                df_results.loc[column_name, 'Rel. Perf.'].loc['F1-score', 'Control'] = 1 - f1_score_cp / f1_score_cn
            else:
                df_results.loc[column_name, 'Positive'].loc['F1-score', 'Control'] = 'no tp'
                df_results.loc[column_name, 'Negative'].loc['F1-score', 'Control'] = 'no tp'
                df_results.loc[column_name, 'Rel. Perf.'].loc['F1-score', 'Control'] = 'no tp'


            acc_real_p = cal_acc_real_fake(df_real_positive['output_label'], df_real_positive['real_label'])
            acc_real_n = cal_acc_real_fake(df_real_negative['output_label'], df_real_negative['real_label'])
            acc_fake_p = cal_acc_real_fake(df_fake_positive['output_label'], df_fake_positive['real_label'])
            acc_fake_n = cal_acc_real_fake(df_fake_negative['output_label'], df_fake_negative['real_label'])
            df_results.loc[column_name, 'Positive'].loc['Err (Real)', 'Data'] = 1 - acc_real_p
            df_results.loc[column_name, 'Negative'].loc['Err (Real)', 'Data'] = 1 - acc_real_n
            if acc_real_n != 1:
                df_results.loc[column_name, 'Rel. Perf.'].loc['Err (Real)', 'Data'] = 1 - (1 - acc_real_p) / (1 - acc_real_n)
            else:
                df_results.loc[column_name, 'Rel. Perf.'].loc['Err (Real)', 'Data'] = 'All correct'
            df_results.loc[column_name, 'Positive'].loc['Err (Fake)', 'Data'] = 1 - acc_fake_p
            df_results.loc[column_name, 'Negative'].loc['Err (Fake)', 'Data'] = 1 - acc_fake_n
            if acc_fake_n != 1:
                df_results.loc[column_name, 'Rel. Perf.'].loc['Err (Fake)', 'Data'] = 1 - (1 - acc_fake_p) / (1 - acc_fake_n)
            else:
                df_results.loc[column_name, 'Rel. Perf.'].loc['Err (Fake)', 'Data'] = 'All correct'

            acc_real_cp = cal_acc_real_fake(df_real_positive_control['output_label'], df_real_positive_control['real_label'])
            acc_real_cn = cal_acc_real_fake(df_real_negative_control['output_label'], df_real_negative_control['real_label'])
            acc_fake_cp = cal_acc_real_fake(df_fake_positive_control['output_label'], df_fake_positive_control['real_label'])
            acc_fake_cn = cal_acc_real_fake(df_fake_negative_control['output_label'], df_fake_negative_control['real_label'])
            df_results.loc[column_name, 'Positive'].loc['Err (Real)', 'Control'] = 1 - acc_real_cp
            df_results.loc[column_name, 'Negative'].loc['Err (Real)', 'Control'] = 1 - acc_real_cn
            if acc_real_cn != 1:
                df_results.loc[column_name, 'Rel. Perf.'].loc['Err (Real)', 'Control'] = 1 - (1 - acc_real_cp) / (
                            1 - acc_real_cn)
            else:
                df_results.loc[column_name, 'Rel. Perf.'].loc['Err (Real)', 'Control'] = 'All correct'
            df_results.loc[column_name, 'Positive'].loc['Err (Fake)', 'Control'] = 1 - acc_fake_cp
            df_results.loc[column_name, 'Negative'].loc['Err (Fake)', 'Control'] = 1 - acc_fake_cn
            if acc_fake_cn != 1:
                df_results.loc[column_name, 'Rel. Perf.'].loc['Err (Fake)', 'Control'] = 1 - (1 - acc_fake_cp) / (
                            1 - acc_fake_cn)
            else:
                df_results.loc[column_name, 'Rel. Perf.'].loc['Err (Fake)', 'Control'] = 'All correct'
        else:
            df_results.loc[column_name, :] = 'No real/fake'
        print(df_results.loc[column_name, 'Rel. Perf.'].loc['Err (Fake)', 'Control'])

    df_results.to_csv(output_file)


def cal_frr_at_far(df_label, df_score, column_name, flag):
    far, tar, thresholds = roc_curve(df_label, df_score)
    fpr, tpr = far, tar
    frr = 1 - tar

    plot_roc(fpr, tpr, column_name, flag)
    tpr_01, tpr_01_thr = compute_tpr(fpr, 0.1, thresholds, tpr)
    tpr_001, tpr_001_thr = compute_tpr(fpr, 0.01, thresholds, tpr)
    tpr_0001, tpr_0001_thr = compute_tpr(fpr, 0.001, thresholds, tpr)
    return tpr_01, tpr_001, tpr_0001


def cal_acc(output_label, real_label):
    tn = tp = fn = fp = 0
    for i in range(len(output_label)):
        if output_label.iloc[i] == 0 and real_label.iloc[i] == 0:
            tn = tn + 1
        elif output_label.iloc[i] != 0 and real_label.iloc[i] != 0:
            tp = tp + 1
        elif output_label.iloc[i] != 0 and real_label.iloc[i] == 0:
            fp = fp + 1
        elif output_label.iloc[i] == 0 and real_label.iloc[i] != 0:
            fn = fn + 1
        else:
            sys.exit('wrong acc calculation')
    print(tn, tp, fn, fp)

    acc = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    balanced_acc = (sensitivity + specificity) / 2
    if tp == 0:
        f1_score = 'tp+fp=0'
    else:
        precision = tp / (tp + fp)
        f1_score = 2 * precision * sensitivity / (precision + sensitivity)

    return acc, balanced_acc, f1_score


def cal_acc_real_fake(output_label, real_label):
    correct = np.equal(output_label, real_label)
    count = np.count_nonzero(correct)
    acc = count/len(output_label)

    return acc


def plot_det(far, frr, plot_name, flag):
    if flag == 1:
        title = plot_name + '_positive'
    elif flag == 2:
        title = plot_name + '_negative'
    elif flag == 3:
        title = plot_name + '_control_positive'
    elif flag == 4:
        title = plot_name + '_control_negative'
    else:
        sys.exit('wrong flag')

    plt.figure()
    plt.plot(far, frr)
    plt.title(title)
    plt.xlabel("FAR")
    plt.ylabel("FRR")
    plt.xscale('log')
    plt.ylim([0, 1])
    plt.savefig(f"plots/{dataset}/{title}.png")
    tikzplotlib.save(f"plots/{dataset}/{title}.tex")
    plt.clf()


def plot_roc(fpr, tpr, plot_name, flag):
    if flag == 1:
        title = plot_name + '_positive'
    elif flag == 2:
        title = plot_name + '_negative'
    elif flag == 3:
        title = plot_name + '_control_positive'
    elif flag == 4:
        title = plot_name + '_control_negative'
    else:
        sys.exit('wrong flag')

    plt.figure()
    plt.plot(fpr, tpr)
    plt.title(title)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.xscale('log')
    plt.xlim([0.0005, 1])
    plt.ylim([0, 1])
    if not os.path.exists(f"plots/{dataset}"):
        os.makedirs(f"plots/{dataset}")
    plt.savefig(f"plots/{dataset}/{title}.png")
    tikzplotlib.save(f"plots/{dataset}/{title}.tex")
    plt.clf()


def compute_eer(tar, far, thresholds):
    values = np.abs(1 - tar - far)
    idx = np.argmin(values)
    eer = far[idx]
    thr = thresholds[idx]
    return eer, thr


def compute_frr(far, FAR_threshold, thresholds, frr):
    values = np.abs(far - FAR_threshold)
    idx = np.argmin(values)
    thr = thresholds[idx]
    return frr[idx], thr


def compute_tpr(fpr, FPR_threshold, thresholds, tpr):
    values = np.abs(fpr - FPR_threshold)
    # idx = np.argmin(values)
    idx = np.where(values == values.min())[0][-1]
    # if idx == 0:
    #     idx = 1
    thr = thresholds[idx]
    return tpr[idx], thr


if __name__ == "__main__":
    main()

"""
Plotting of behavioral metrics during the full task (biased blocks) per lab

Guido Meijer
6 May 2020
"""

import seaborn as sns
import numpy as np
from os.path import join
import matplotlib.pyplot as plt
from scipy import stats
import scikit_posthocs as sp
from paper_behavior_functions import (figpath, seaborn_style, group_colors, institution_map,
                                      FIGURE_WIDTH, FIGURE_HEIGHT, QUERY,
                                      fit_psychfunc, dj2pandas, load_csv)
import pandas as pd
from statsmodels.stats.multitest import multipletests

# Initialize
seaborn_style()
figpath = figpath()
pal = group_colors()
institution_map, col_names = institution_map()
col_names = col_names[:-1]

# %% Process data

if QUERY is True:
    # query sessions
    from paper_behavior_functions import query_sessions_around_criterion
    from ibl_pipeline import reference, subject, behavior
    use_sessions, _ = query_sessions_around_criterion(criterion='ephys',
                                                      days_from_criterion=[2, 0],
                                                      force_cutoff=True)
    session_keys = (use_sessions & 'task_protocol LIKE "%biased%"').fetch('KEY')
    ses = ((use_sessions & 'task_protocol LIKE "%biased%"')
           * subject.Subject * subject.SubjectLab * reference.Lab
           * (behavior.TrialSet.Trial & session_keys))
    ses = ses.proj('institution_short', 'subject_nickname', 'task_protocol', 'session_uuid',
                   'trial_stim_contrast_left', 'trial_stim_contrast_right',
                   'trial_response_choice', 'task_protocol', 'trial_stim_prob_left',
                   'trial_feedback_type', 'trial_response_time', 'trial_stim_on_time',
                   'session_end_time').fetch(
                       order_by='institution_short, subject_nickname,session_start_time, trial_id',
                       format='frame').reset_index()
    behav = dj2pandas(ses)
    behav['institution_code'] = behav.institution_short.map(institution_map)
else:
    behav = load_csv('Fig4.csv')

biased_fits = pd.DataFrame()
for i, nickname in enumerate(behav['subject_nickname'].unique()):
    if np.mod(i+1, 10) == 0:
        print('Processing data of subject %d of %d' % (i+1,
                                                       len(behav['subject_nickname'].unique())))

    # Get lab
    lab = behav.loc[behav['subject_nickname'] == nickname, 'institution_code'].unique()[0]

    # Fit psychometric curve
    left_fit = fit_psychfunc(behav[(behav['subject_nickname'] == nickname)
                                   & (behav['probabilityLeft'] == 80)])
    right_fit = fit_psychfunc(behav[(behav['subject_nickname'] == nickname)
                                    & (behav['probabilityLeft'] == 20)])
    fits = pd.DataFrame(data={'threshold_l': left_fit['threshold'],
                              'threshold_r': right_fit['threshold'],
                              'bias_l': left_fit['bias'],
                              'bias_r': right_fit['bias'],
                              'lapselow_l': left_fit['lapselow'],
                              'lapselow_r': right_fit['lapselow'],
                              'lapsehigh_l': left_fit['lapsehigh'],
                              'lapsehigh_r': right_fit['lapsehigh'],
                              'nickname': nickname, 'lab': lab})
    biased_fits = biased_fits.append(fits, sort=False)

# %% Statistics
stats_tests = pd.DataFrame(columns=['variable', 'test_type', 'p_value'])
posthoc_tests = {}

for i, var in enumerate(['threshold_l', 'threshold_r', 'lapselow_l', 'lapselow_r', 'lapsehigh_l',
                         'lapsehigh_r', 'bias_l', 'bias_r']):
    _, normal = stats.normaltest(biased_fits[var])

    if normal < 0.05:
        test_type = 'kruskal'
        test = stats.kruskal(*[group[var].values
                               for name, group in biased_fits.groupby('lab')])
        if test[1] < 0.05:  # Proceed to posthocs
            posthoc = sp.posthoc_dunn(biased_fits, val_col=var, group_col='lab')
        else:
            posthoc = np.nan
    else:
        test_type = 'anova'
        test = stats.f_oneway(*[group[var].values
                                for name, group in biased_fits.groupby('lab')])
        if test[1] < 0.05:
            posthoc = sp.posthoc_tukey(biased_fits, val_col=var, group_col='lab')
        else:
            posthoc = np.nan

    posthoc_tests['posthoc_'+str(var)] = posthoc
    stats_tests.loc[i, 'variable'] = var
    stats_tests.loc[i, 'test_type'] = test_type
    stats_tests.loc[i, 'p_value'] = test[1]

# Correct for multiple tests
stats_tests['p_value'] = multipletests(stats_tests['p_value'], method='fdr_bh')[1]

# Test between left/right blocks
for i, var in enumerate(['threshold', 'lapselow', 'lapsehigh', 'bias']):
    stats_tests.loc[stats_tests.shape[0] + 1, 'variable'] = '%s_blocks' % var
    stats_tests.loc[stats_tests.shape[0], 'test_type'] = 'wilcoxon'
    _, stats_tests.loc[stats_tests.shape[0], 'p_value'] = stats.wilcoxon(
                                    biased_fits['%s_l' % var], biased_fits['%s_r' % var])
print(stats_tests)  # Print the results

# %% Plot metrics
f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(FIGURE_WIDTH*0.8, FIGURE_HEIGHT))
lab_colors = group_colors()

ax1.plot([10, 20], [10, 20], linestyle='dashed', color=[0.6, 0.6, 0.6])
for i, lab in enumerate(biased_fits['lab'].unique()):
    ax1.errorbar(biased_fits.loc[biased_fits['lab'] == lab, 'threshold_l'].mean(),
                 biased_fits.loc[biased_fits['lab'] == lab, 'threshold_r'].mean(),
                 xerr=biased_fits.loc[biased_fits['lab'] == lab, 'threshold_l'].sem(),
                 yerr=biased_fits.loc[biased_fits['lab'] == lab, 'threshold_l'].sem(),
                 fmt='.', color=lab_colors[i])
ax1.set(xlabel='80:20 block', ylabel='20:80 block', title='Threshold',
        yticks=ax1.get_xticks(), ylim=ax1.get_xlim())

ax2.plot([0, 0.1], [0, 0.1], linestyle='dashed', color=[0.6, 0.6, 0.6])
for i, lab in enumerate(biased_fits['lab'].unique()):
    ax2.errorbar(biased_fits.loc[biased_fits['lab'] == lab, 'lapselow_l'].mean(),
                 biased_fits.loc[biased_fits['lab'] == lab, 'lapselow_r'].mean(),
                 xerr=biased_fits.loc[biased_fits['lab'] == lab, 'lapselow_l'].sem(),
                 yerr=biased_fits.loc[biased_fits['lab'] == lab, 'lapselow_r'].sem(),
                 fmt='.', color=lab_colors[i])
ax2.set(xlabel='80:20 block', ylabel='', title='Lapse left',
        yticks=ax2.get_xticks(), ylim=ax2.get_xlim())

ax3.plot([0, 0.1], [0, 0.1], linestyle='dashed', color=[0.6, 0.6, 0.6])
for i, lab in enumerate(biased_fits['lab'].unique()):
    ax3.errorbar(biased_fits.loc[biased_fits['lab'] == lab, 'lapsehigh_l'].mean(),
                 biased_fits.loc[biased_fits['lab'] == lab, 'lapsehigh_r'].mean(),
                 xerr=biased_fits.loc[biased_fits['lab'] == lab, 'lapsehigh_l'].sem(),
                 yerr=biased_fits.loc[biased_fits['lab'] == lab, 'lapsehigh_l'].sem(),
                 fmt='.', color=lab_colors[i])
ax3.set(xlabel='80:20 block', ylabel='', title='Lapse right',
        yticks=ax3.get_xticks(), ylim=ax3.get_xlim())

ax4.plot([-10, 10], [-10, 10], linestyle='dashed', color=[0.6, 0.6, 0.6])
for i, lab in enumerate(biased_fits['lab'].unique()):
    ax4.errorbar(biased_fits.loc[biased_fits['lab'] == lab, 'bias_l'].mean(),
                 biased_fits.loc[biased_fits['lab'] == lab, 'bias_r'].mean(),
                 xerr=biased_fits.loc[biased_fits['lab'] == lab, 'bias_l'].sem(),
                 yerr=biased_fits.loc[biased_fits['lab'] == lab, 'bias_l'].sem(),
                 fmt='.', color=lab_colors[i])
ax4.set(xlabel='80:20 block', ylabel='', title='Bias',
        yticks=ax4.get_xticks(), ylim=ax4.get_xlim())

plt.tight_layout(w_pad=-0.1)
sns.despine(trim=True)
plt.savefig(join(figpath, 'figure4f-i_metrics_per_lab_full.pdf'))
plt.savefig(join(figpath, 'figure4f-i_metrics_per_lab_full.png'), dpi=300)

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# parameters of the experiment
sessions = ['G', 'P60', 'P80', 'GP60', 'GP80', 'I60', 'I80']
xlimits_mm = [0, 476.06]
ylimits_mm = [0, 267.79]
xlimits_px = [0, 2400]
ylimits_px = [0, 1350]


def read_data(name_dir):

    df = pd.concat([
        pd.read_csv(os.path.join(name_dir, file)).assign(participant_ID=file.split('.')[0])
        for file in os.listdir(name_dir) if file.endswith('.txt')
    ], ignore_index=True)
    df.rename(columns={'trajectory id[1-7]': 'trajectory'}, inplace=True)
    df['condition'] = ''
    conditions_map = {
        ('G', 0): 'G', ('P', 60): 'P60', ('P', 80): 'P80',
        ('GP', 60): 'GP60', ('GP', 80): 'GP80',
        ('I', 60): 'I60', ('I', 80): 'I80'
    }
    df['condition'] = df.apply(lambda row: conditions_map.get((row['mode[str]'], row['percentage']), ''), axis=1)

    return df

def x_px_to_mm(px, total_px=xlimits_px[1], total_mm=xlimits_mm[1]):
    return round((px * total_mm) / total_px, 2)

def y_px_to_mm(px, total_px=ylimits_px[1], total_mm=ylimits_mm[1]):
    return round((px * total_mm) / total_px, 2)

def set_to_mm():

    df['goal point x[mm]'] = x_px_to_mm(df['goal point x[px]'])
    df['goal point y[mm]'] = y_px_to_mm(df['goal point y[px]'])
    df['guessed point x[mm]'] = x_px_to_mm(df['guessed point x[px]'])
    df['guessed point y[mm]'] = y_px_to_mm(df['guessed point y[px]'])

def compute_bias():

    df['bias_x'] = df['guessed point x[px]'] - df['goal point x[px]']
    df['bias_y'] = df['guessed point y[px]'] - df['goal point y[px]']

    df['bias_x'] = df['bias_x'].apply(lambda x: round(abs(x_px_to_mm(x)), 2))
    df['bias_y'] = df['bias_y'].apply(lambda y: round(abs(y_px_to_mm(y)), 2))

    df['bias_tot'] = np.sqrt(df['bias_x'] ** 2 + df['bias_y'] ** 2).round(2)


def create_mixed_model_format_df():

    df.to_csv('all.csv', index=False)

    participants_df = df.groupby(['participant_ID', 'condition']).mean(numeric_only=True).reset_index()
    participants_df.to_csv('participants.csv', sep='\t', index=False)

    avg_df = df.groupby(['condition']).mean(numeric_only=True).reset_index()
    avg_df.to_csv('avg.csv', sep='\t', index=False, )

    trajectories_df = df.groupby(['condition', 'trajectory']).mean(numeric_only=True).reset_index()
    trajectories_df['coordinates'] = trajectories_df.apply(lambda row: np.array([row['guessed point x[mm]'], row['guessed point y[mm]']]), axis=1)
    trajectories_df = compute_centroid(trajectories_df, avg_participants=True)
    trajectories_df.to_csv('trajectories.csv', sep='\t', index=False)

    mixed_model_df = df.groupby(['participant_ID', 'condition', 'trajectory']).mean(numeric_only=True).reset_index()
    mixed_model_df['coordinates'] = mixed_model_df.apply(lambda row: np.array([row['guessed point x[mm]'], row['guessed point y[mm]']]), axis=1)
    mixed_model_df = compute_centroid(mixed_model_df, avg_participants=False)
    mixed_model_df.to_csv('mixed_model.csv', sep='\t', index=False)

    std_df = df.groupby(['participant_ID', 'condition', 'trajectory']).std(numeric_only=True).reset_index()
    std_df.to_csv('std.csv', sep='\t', index=False)

    return participants_df, avg_df, trajectories_df, mixed_model_df, std_df

def save_ttest_df(df, parameter, aggfunc):
    columns_to_keep = ['participant_ID', 'trajectory']
    columns_to_keep += df['condition'].unique().tolist()
    paired_samples_df = df.pivot_table(
        index=['participant_ID', 'trajectory'],  # Each row for a participant
        columns='condition',  # Columns for each condition
        values=parameter,  # Values to aggregate
        aggfunc=aggfunc  # Aggregation function
    ).reset_index()
    paired_samples_df.columns.name = None
    paired_samples_df[columns_to_keep].to_csv(f'paired_samples_traj_{parameter}.csv', sep='\t', index=False)

    paired_samples_df = df.pivot_table(
        index=['participant_ID'],  # Each row for a participant
        columns='condition',  # Columns for each condition
        values=parameter,  # Values to aggregate
        aggfunc=aggfunc  # Aggregation function
    ).reset_index()
    paired_samples_df.columns.name = None
    columns_to_keep.remove('trajectory')
    paired_samples_df[columns_to_keep].to_csv(f'paired_samples_avg_{parameter}.csv', sep='\t', index=False)

def create_ttest_format_df():
    
    save_ttest_df(df, parameter='bias_tot', aggfunc='mean')
    save_ttest_df(df, parameter='bias_x', aggfunc='mean')
    save_ttest_df(df, parameter='bias_y', aggfunc='mean')

    save_ttest_df(mixed_model_df, parameter='centroid_distance', aggfunc='mean')
    save_ttest_df(mixed_model_df, parameter='centroid_delta_x', aggfunc='mean')
    save_ttest_df(mixed_model_df, parameter='centroid_delta_y', aggfunc='mean')

    save_ttest_df(df, parameter='guessed point x[mm]', aggfunc='std')
    save_ttest_df(df, parameter='guessed point y[mm]', aggfunc='std')

def compute_centroid(df, avg_participants=False):

    if avg_participants:
        variables = ['condition']
    else:
        variables = ['participant_ID', 'condition']
    centroid_df = df.groupby(variables).agg({'coordinates': 'mean'}).reset_index()
    centroid_df.columns.name = None
    centroid_df.rename(columns={'coordinates': 'centroid_coord'}, inplace=True)
    df = pd.merge(df, centroid_df, on=variables, how='left')
    df['centroid_distance'] = df.apply(lambda row: np.linalg.norm(row['coordinates'] - row['centroid_coord']), axis=1)
    df['centroid_delta_x'] = df.apply(lambda row: abs(row['coordinates'][0] - row['centroid_coord'][0]), axis=1)
    df['centroid_delta_y'] = df.apply(lambda row: abs(row['coordinates'][1] - row['centroid_coord'][1]), axis=1)

    return df


def plot(title, condition_list):

    responses_df = trajectories_df[trajectories_df['condition'].isin(condition_list)]

    sns.scatterplot(x='goal point x[mm]', y='goal point y[mm]', data=responses_df, style=responses_df['trajectory'],
                    color='#808080', s=150, legend=False)
    palette = 'colorblind'
    ax = sns.scatterplot(x='guessed point x[mm]', y='guessed point y[mm]', data=responses_df,
                         hue=responses_df['condition'], style=responses_df['trajectory'], s=150, palette=palette)

    handles, labels = ax.get_legend_handles_labels()
    stimuli_handle = plt.scatter([], [], color='#808080', marker='o', s=70, label='stimuli')
    handles = [stimuli_handle] + handles
    labels = ['stimuli'] + labels
    plt.xlim(xlimits_mm[0], xlimits_mm[1])
    plt.ylim(ylimits_mm[0], ylimits_mm[1])
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')

    plt.legend(bbox_to_anchor=(0.75, 0.95), loc='upper left', handles=handles, labels=labels)
    for label in ax.legend_.get_texts():
        if label.get_text() in ['stimuli', 'condition', 'trajectory']:
            label.set_fontweight('bold')
    plt.title(title)
    #plt.show()
    plt.savefig(f"{title}.png")
    plt.close()

def plot_participants(title, condition_list):

    responses_df = trajectories_df[trajectories_df['condition'].isin(condition_list)]
    all_responses_df = mixed_model_df[mixed_model_df['condition'].isin(condition_list)]

    sns.scatterplot(x='goal point x[mm]', y='goal point y[mm]', data=responses_df, style=responses_df['trajectory'],
                    color='#808080', s=150, legend=False)
    palette = 'colorblind'
    ax = sns.scatterplot(x='guessed point x[mm]', y='guessed point y[mm]', data=responses_df,
                         hue=responses_df['condition'], style=responses_df['trajectory'], s=150, palette=palette)

    sns.scatterplot(x='guessed point x[mm]', y='guessed point y[mm]', data=all_responses_df, hue=all_responses_df['condition'],
                    style=all_responses_df['trajectory'], s=30, palette=palette, legend=False)

    handles, labels = ax.get_legend_handles_labels()
    stimuli_handle = plt.scatter([], [], color='#808080', marker='o', s=70, label='stimuli')
    handles = [stimuli_handle] + handles
    labels = ['stimuli'] + labels
    plt.xlim(xlimits_mm[0], xlimits_mm[1])
    plt.ylim(ylimits_mm[0], ylimits_mm[1])
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')

    plt.legend(bbox_to_anchor=(0.75, 0.95), loc='upper left', handles=handles, labels=labels)
    for label in ax.legend_.get_texts():
        if label.get_text() in ['stimuli', 'condition', 'trajectory']:
            label.set_fontweight('bold')
    plt.title(title)
    #plt.show()
    plt.savefig(f"{title}.png")
    plt.close()


if __name__ == '__main__':

    df = read_data('data')
    set_to_mm()
    compute_bias()

    participants_df, avg_df, trajectories_df, mixed_model_df, std_df = create_mixed_model_format_df()
    create_ttest_format_df()

    plot(title='Progressive Information in Trajectories completion (P)', condition_list=['G', 'P60', 'P80'])
    plot(title='Progressive Information in Trajectories completion (GP)', condition_list=['G', 'GP60', 'GP80'])
    plot(title= 'Integration of Gaze in 60 percent completion', condition_list=['G', 'P60', 'GP60'])
    plot(title='Integration of Gaze in 80 percent completion', condition_list=['G', 'P80', 'GP80'])
    plot(title='Gaze Coherent and Incoherent in 60 percent completion', condition_list=['GP60', 'I60'])
    plot(title='Gaze Coherent and Incoherent in 80 percent completion', condition_list=['GP80', 'I80'])

    plot_participants(title='Example Plot with Participants Data for G condition' , condition_list=['G'])
    print("all plots done")
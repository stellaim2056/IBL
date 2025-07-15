import numpy as np

def get_task_type(trials_df, print_info=False):
        if 'probabilityLeft' in trials_df.columns:
            unique_prob = np.unique(trials_df['probabilityLeft'].dropna())
            if len(unique_prob) == 1 and np.isclose(unique_prob[0], 0.5):
                if print_info:
                    print("Identified as a basic task.")
                return 'basic'
            else:
                if print_info:
                    print("Identified as a full task.")
                return 'full'
        else:
            if print_info:
                print("Task type is unknown.")
            return 'unknown'
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def spatial_train_test_split(train_df: pd.DataFrame, 
                             x_col: str, y_col: str,
                             test_size: float = 0.2,
                             seed: int = 42):
    '''
    Takes pd.DataFrame. Assigns a part of (x, y) spatial units to train,
        the rest to the test.
        This information will be reused to split train_df to train and test_spatial (same date, diff xy),
        and split test_temp to test_temp (same xy, diff date) and test_temp_spatial (diff date, diff xy)
    '''
    unique_coordinates = np.unique(train_df[[x_col, y_col]], axis=0)
    train_coors, test_coors = train_test_split(unique_coordinates, test_size=test_size, random_state=seed)

    train_coors_df = pd.DataFrame(train_coors, columns=[x_col, y_col])
    test_coors_df = pd.DataFrame(test_coors, columns=[x_col, y_col])

    return train_coors_df, test_coors_df

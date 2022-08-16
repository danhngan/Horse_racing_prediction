import pandas as pd
import numpy as np
from collections import deque


def split_data(X, y, raceid: pd.Series, top3:pd.Series, drop_columns = None, select_cols = None, test_size = 0.3):
    """
    `X_train, X_test, y_train, y_test, top3_train, top3_test, raceid_train, raceid_test` according to `test_size`
    
    if `test_size` is float and less than 1, test dataset will have (number of group)*test_size (round down) races.
    
    if `test_size` is int and less than `df.shape[0]`, test dataset will have test_size instances"""
    

    if isinstance(test_size, float) and test_size < 1:
        # Group by race id
        race_groups = raceid.groupby(lambda x: raceid[x])
        
        # Find separated index
        n_groups = race_groups.ngroups
        n_train_groups = (1-test_size)*n_groups
        group_idx = race_groups.ngroup()
        idx = np.searchsorted(group_idx, n_train_groups, side='left')
        sep_group_name = raceid.iloc[idx]
        sep_idx = race_groups.groups[sep_group_name][-1]

    elif isinstance(test_size, int) and test_size < raceid.shape[0]:
        sep_idx = raceid.shape[0] - test_size - 1
    

    # Split data
    if drop_columns is not None:
        X = X.reset_index(drop=True).drop([col for col in X.columns if col in drop_columns], axis=1)
    if select_cols is None:
        select_cols = X.columns
    X_train = X.loc[:sep_idx, select_cols]
    X_test = X.loc[sep_idx+1:, select_cols]

    y_train = y.iloc[:sep_idx+1]
    y_test = y.iloc[sep_idx+1:]

    top3_train = top3.iloc[:sep_idx+1]
    top3_test = top3.iloc[sep_idx+1:]

    raceid_train = raceid.iloc[:sep_idx+1]
    raceid_test = raceid.iloc[sep_idx+1:]

    return X_train, X_test, y_train, y_test, top3_train, top3_test, raceid_train, raceid_test



def top3_time(time_group: pd.Series):
    """return indexes of the top 3 horses in race"""
    return time_group.sort_values().index[:3]

def top3_prob(prob_group: pd.Series):
    """return the top 3 highest probabilities"""
    return prob_group.sort_values(ascending=False).index[:3]

def predict_top3_time(model, X, raceid):
    """return a DataFrame that indicates the top 3 of race according to predictive time"""

    assert len(X) == len(raceid), 'Length of X is not equal to length of raceid'
    y_pred = model.predict(X)
    df_top3 = pd.DataFrame({'raceid': raceid, 'Time': y_pred})
    df_top3['Top 3'] = False
    df_group = df_top3.groupby('raceid')
    for group in df_group.groups:
        df_top3.loc[top3_time(df_group.get_group(group)['Time']), 'Top 3'] = True
    return df_top3

def predict_top3_prob(model, X, raceid):
    """return a DataFrame that indicates the top 3 of race according to predictive probabilities"""
    assert len(X) == len(raceid), 'Length of X is not equal to length of raceid'
    if hasattr(model, 'predict_proba'):
        y_pred = model.predict_proba(X)[:,1]
    else:
        y_pred = model.predict(X).flatten()
    df_top3 = pd.DataFrame({'raceid': raceid, 'Top 3 prob': y_pred})
    df_top3['Top 3'] = False
    df_group = df_top3.groupby('raceid')
    for group in df_group.groups:
        df_top3.loc[top3_prob(df_group.get_group(group)['Top 3 prob']), 'Top 3'] = True
    return df_top3

class generate_exp():
    def __init__(self, colname, val = None, k = 0, init_val=0.):
        self.exp_dict = {}
        self.k = k
        self.init_val = init_val
        self.colname = colname
        self.val = val

    
    def generate_exp(self, row):
        if row[self.colname] in self.exp_dict:
            ans = self.exp_dict[row[self.colname]]
            if self.val is None:
                self.exp_dict[row[self.colname]] += 1.
            else:
                self.exp_dict[row[self.colname]] += float(row[self.val])
        else:
            if self.val is None:
                self.exp_dict[row[self.colname]] = 1.
                ans = self.init_val
            else:
                self.exp_dict[row[self.colname]] = float(row[self.val])
                ans = self.init_val
        return ans
    

    def generate_strike(self, row):
        if self.val is None:
            raise ValueError('This function does nothing without val!')
        
        if row[self.colname] in self.exp_dict:
            ans = self.exp_dict[row[self.colname]]
            if bool(row[self.val]):
                self.exp_dict[row[self.colname]] += 1.
            else:
                self.exp_dict[row[self.colname]] = 0.
        else:
            self.exp_dict[row[self.colname]] = float(row[self.val])
            ans = self.init_val
        return ans
    

    def generate_last_k_exp(self, row):
        
        if self.val is None:
            raise ValueError('This function does nothing without val!')

        new_key = row[self.colname]
        new_val = float(row[self.val])

        if new_key in self.exp_dict:

            self.exp_dict[new_key][0] += self.exp_dict[new_key][1][-1] - self.exp_dict[new_key][1][0]
            self.exp_dict[new_key][1].append(new_val)
            self.exp_dict[new_key][1].popleft()
                           
        else:
            temp = deque([self.init_val/self.k]*self.k)
            temp.append(new_val)
            self.exp_dict[new_key] = [self.init_val, temp]

        return self.exp_dict[new_key][0]
    
    def generate_last_k_var(self, row):
        """
        """
        if self.val is None:
            raise ValueError('This function does nothing without val!')

        self.var_exp = {}

        new_key = row[self.colname]
        new_val = float(row[self.val])
        if  new_key in self.exp_dict:

            pvar = self.var_exp[new_key][-1]
            pmean = self.exp_dict[new_key][0]/self.k
            pe2 = pvar + pmean**2

            self.exp_dict[new_key][0] += self.exp_dict[new_key][1][-1] - self.exp_dict[new_key][1][0]
            
            nmean = self.exp_dict[new_key][0]/self.k
            ne2 = pe2 + ((self.exp_dict[new_key][1][-1]/self.k)**2 - (self.exp_dict[new_key][1][0]/self.k)**2)/self.k
            self.var_exp[new_key].append(ne2 - nmean**2)

            self.exp_dict[new_key][1].append(new_val)
            self.exp_dict[new_key][1].popleft()
                           
        else:
            temp = deque([self.init_val/self.k]*self.k)
            temp.append(new_val)
            self.exp_dict[new_key] = [self.init_val, temp]
            self.var_exp[new_key] = [0]

        return self.exp_dict[new_key][0], self.var_exp[new_key]
import warnings
import pandas as pd
import numpy as np

def multiclass_prediction_from_probabilities(probabilities:pd.DataFrame,
                                             cutoffs:dict,
                                             show_warning=False):
    """

    :param probabilities: A DataFrame typically build from cls.predict_proba output.
    :param cutoffs: A dictionary with cutoff values for each column of probabilities.
    :param show_warning: Logical to silence the function-call.
    :return: The predictions as classes.
    """

    cols = probabilities.columns

    cutoff_keys = set(cutoffs.keys())
    cols_set = set(cols)

    if cutoff_keys != cols_set:
        raise KeyError('Cutoff-Keys and column-names of prababilities DataFrame must be equal.')

    # Translate probabilities into classes
    manual_preds_0 = [cols[0] if value > cutoffs[cols[0]] else np.nan for value in probabilities.loc[:,cols[0]]]
    manual_preds_1 = [cols[1] if value > cutoffs[cols[1]] else np.nan for value in probabilities.loc[:, cols[1]]]
    manual_preds_2 = [cols[2] if value > cutoffs[cols[2]] else np.nan for value in probabilities.loc[:, cols[2]]]

    manual_preds_df = pd.DataFrame({cols[0]: manual_preds_0,
                                    cols[1]: manual_preds_1,
                                    cols[2]: manual_preds_2})
    manual_preds_df['0'] = manual_preds_df['0'].astype('float64')
    manual_preds_df['1'] = manual_preds_df['1'].astype('float64')
    manual_preds_df['2'] = manual_preds_df['2'].astype('float64')

    df_dicho = manual_preds_df.notnull()
    n_preds = df_dicho.apply(sum, axis=1)

    # Find out where we have >=2 predictions
    dup_preds_locations = np.where(n_preds >= 2)[0]

    for i in dup_preds_locations:
        dicho_row = df_dicho.iloc[i,:]
        preds_proba_row = probabilities.iloc[i,:]
        preds_proba_row.index = dicho_row.index

        max_value = preds_proba_row[dicho_row].max()

        # Check for duplicated max_value
        if sum(preds_proba_row == max_value) > 1:
            if show_warning:
                warnings.warn("Duplicated probability value - no decision can be made for row "+ str(i))
            manual_preds_df.loc[i, :] = np.nan
        else:
            mask = preds_proba_row != max_value
            manual_preds_df.loc[i,mask] = np.nan

    out = pd.Series([np.nan]*len(probabilities), index=manual_preds_df.index, dtype='object')
    for row in manual_preds_df.iterrows():
        row_mask = ~row[1].isnull()
        if sum(row_mask) == 1:
            out.iloc[row[0]] = row[1][row_mask].iloc[0]
        else:
            out.iloc[row[0]] = np.nan

    out = out.astype('float64')
    return out.values
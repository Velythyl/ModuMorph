import numpy as np


def box_and_whisker_stats(avg_score):
    data = np.array(avg_score)
    q1 = np.percentile(data, 25)
    q2 = np.percentile(data, 50)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1

    lower_whisker = max(data[data >= q1 - 1.5 * iqr].min(), data.min())
    upper_whisker = min(data[data <= q3 + 1.5 * iqr].max(), data.max())

    #outliers = data[(data < lower_whisker) | (data > upper_whisker)]

    return {
        "mean": data.mean(),
        "std": data.std(),
        "median": q2,
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower_whisker": lower_whisker,
        "upper_whisker": upper_whisker,
        #"outliers": outliers
    }
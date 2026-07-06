from torres.m1 import pair_input_output
from sklearn.linear_model import LinearRegression
import numpy as np
from torres.stats import mae, x_axis_error, lag_ln10
from torres.m1 import evaluate

def run_linear_baseline(data, prediction_time):

    train, targets_train, test, targets_test = pair_input_output(data, False, prediction_time)

    X_train = np.array([instance.flatten() for instance in train])
    X_test = np.array([instance.flatten() for instance in test])

    lr_model = LinearRegression()
    lr_model.fit(X_train, targets_train)
    y_pred = lr_model.predict(X_test)


# ### FROM TORRES MAIN FUNCTION ###
#     # Load events for evaluation



    # _,_,_, targets_test = pair_input_output(data, False, prediction_time)
    event_file = open('data/event_timestamps.txt', 'r')
    lines = event_file.readlines()
    event_times = [line.split() for line in lines]
    event_file.close()

    # Select only events which are in the test set
    event_times_test = []
    first_test_event_time = list(targets_test.keys())[0]
    for event in event_times:
        if event[0] >= first_test_event_time:
            event_times_test.append(event)

    # Evaluate predictions
    target_times = list(targets_test.keys())
    predictions = {target_times[i]: y_pred[i] for i in range(len(y_pred))}



    evaluate(targets_test, predictions, event_times_test, data, path = f"results/linear/t={prediction_time}", display= False)


    return y_pred, targets_test
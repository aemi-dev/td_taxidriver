The competition dataset is based on the 2016 NYC Yellow Cab trip record data made available in Big Query on Google Cloud Platform. The data was originally published by the NYC Taxi and Limousine Commission (TLC). The data was sampled and cleaned for the purposes of this playground competition. Based on individual trip attributes, participants should predict the duration of each trip in the test set.

File descriptions

train.csv - the training set (contains 1458644 trip records)
test.csv - the testing set (contains 625134 trip records)
sample_submission.csv - a sample submission file in the correct format
Data fields
0 - id - a unique identifier for each trip
1 - vendor_id - a code indicating the provider associated with the trip record
2 - pickup_datetime - date and time when the meter was engaged
3 - dropoff_datetime - date and time when the meter was disengaged
4 - passenger_count - the number of passengers in the vehicle (driver entered value)
5 - pickup_longitude - the longitude where the meter was engaged
6 - pickup_latitude - the latitude where the meter was engaged
7 - dropoff_longitude - the longitude where the meter was disengaged
8 - dropoff_latitude - the latitude where the meter was disengaged
9 - store_and_fwd_flag - This flag indicates whether the trip record was held in vehicle memory before sending to the vendor because the vehicle did not have a connection to the server - Y=store and forward; N=not a store and forward trip
10 - trip_duration - duration of the trip in seconds
Disclaimer: The decision was made to not remove dropoff coordinates from the dataset order to provide an expanded set of variables to use in Kernels.
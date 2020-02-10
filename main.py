import csv

# Variables d'index
id = 0
vendor_id = 1
pickup_datetime = 2
dropoff_datetime = 3
passenger_count = 4
pickup_longitude = 5
pickup_latitude = 6
dropoff_longitude = 7
dropoff_latitude = 8
store_and_fwd_flag = 9
trip_duration = 10

with open('train.csv') as csvfile:
	reader = csv.reader(csvfile)
	next(reader) #Skipping first line

	averagePassengerCount	= 0
	averageTripDuration		= 0

	for row in reader:
		averagePassengerCount	+= int(row[passenger_count])
		averageTripDuration		+= int(row[trip_duration])

		# On affiche les trajets qui ont une durée de plus de 24h
		if int(row[trip_duration]) >= 24*3600:
			seconds	= int(row[trip_duration])

			days	= int(seconds / (24*3600))
			seconds -= days * 24*3600

			hours	= int(seconds / 3600)
			seconds	-= hours * 3600

			mins	= int(seconds / 60)
			seconds -= mins * 60

			print(reader.line_num, ": ", row)
			print(days, "d ", hours, ":", mins, ":", seconds, sep='')
	
	#line_num contient le numéro de la ligne lu. Comme la première ligne ne compte pas (elle contient le nom des colonnes)
	#on utilise line_num - 1 même si sur 1M de lignes ça ne fait pas une grande différence
	averagePassengerCount	/= reader.line_num - 1
	averageTripDuration		/= reader.line_num - 1

	print("Average passenger count:", averagePassengerCount)
	print("Average trip duration: ", averageTripDuration, "s", sep='')

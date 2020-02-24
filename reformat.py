import csv
import math
import time
import datetime

# 0 : id
# 1 : vendorId
# 2 : pickup_datetime
# 3 : dropoff_datetime
# 4 : passenger_count
# 5 : pickup_longitude
# 6 : pickup_latitude
# 7 : dropoff_longitude
# 8 : dropoff_latitude
# 9 : store_and_fwd_flag
# 10 : trip_duration


### Set Constant for better understanding
ID = 0                  # A ENLEVER
VENDORID = 1            # A ENLEVER
PICKUP_DATETIME = 2 
DROPOFF_DATETIME = 3    # A ENLEVER
PASSENGER_COUNT = 4 
PICKUP_LONGITUDE = 5 
PICKUP_LATITUDE = 6 
DROPOFF_LONGITUDE = 7 
DROPOFF_LATITUDE = 8 
STORE_AND_FWD_FLAG = 9  # A ENLEVER
TRIP_DURATION = 10 


def degrees( number ):
    return degrees * math.pi / 180


class Item: 
    def __init__( self, pickup_datetime, trip_duration, passenger_count, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude ):
        self.pickup_datetime = self.time( pickup_datetime )
        self.trip_duration = trip_duration
        self.passenger_count = passenger_count
        self.pickup_longitude = pickup_longitude
        self.pickup_latitude = pickup_latitude
        self.dropoff_longitude = dropoff_longitude;
        self.dropoff_latitude = dropoff_latitude;
        self.trdi = self.distance()

    def pudt( self ):
        return self.pickup_datetime
    def trdu( self ):
        return self.trip_duration
    def paco( self ):
        return self.passenger_count
    def pulg( self ):
        return self.pickup_longitude
    def pult( self ):
        return self.pickup_latitude
    def dolg( self ):
        return self.dropoff_longitude
    def dolt( self ):
        return self.dropoff_latitude
    
    def time( self, string ):
        # +7200 = +2hours because of utc-time difference
        return int( time.mktime(datetime.datetime.strptime( string, "%Y-%m-%d %H:%M:%S").timetuple()) ) + 7200

    def distance( self ):
        R = 6372800  # Earth radius in meters
        lat1, lon1 = float(self.pult()), float(self.pulg())
        lat2, lon2 = float(self.dolt()), float(self.dolg())
        phi1, phi2 = math.radians(lat1), math.radians(lat2) 
        dphi       = math.radians(lat2 - lat1)
        dlambda    = math.radians(lon2 - lon1)
        a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
        return 2*R*math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    def iterable( self ):
        return [self.pudt(),self.trdu(),self.trdi,self.paco(),self.pulg(),self.pult(),self.dolg(),self.dolt()]

    def __str__(self):
        return "['pudt':'{1}','trdu':'{2}','trdi':'{3}','paco':'{4}','pulg':'{5}','pult':'{6}','dolg':'{7}','dolt':'{8}']".format( self.pudt(), self.trdu(), self.trdi, self.paco(), self.pulg(), self.pult(), self.dolg(), self.dolt() )

def readFile ( string ):
    out = open( 'reformat.{}'.format( string ), 'w' )
    out.truncate(0)
    writer = csv.writer( out )
    with open( string, 'r') as f:
        reader = csv.reader(f)
        next( reader )
        writer.writerow( [ 'pickup_datetime', 'trip_duration', 'trip_distance', 'passenger_count', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude' ] )
        for row in reader:
            newItem = Item( row[PICKUP_DATETIME], row[TRIP_DURATION], row[PASSENGER_COUNT], row[PICKUP_LONGITUDE], row[PICKUP_LATITUDE], row[DROPOFF_LONGITUDE], row[DROPOFF_LATITUDE] )
            writer.writerow( newItem.iterable() )

readFile( 'train.csv' )
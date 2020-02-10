import csv
import math
import time
import datetime


def degrees( number ):
    return degrees * math.pi / 180


class Dataset:
    def __init__( self ):
        self.__map = {}
    
    def set( self, key, value ):
        try:
            self.__map[ key ] = value
            return True
        except:
            return False
    
    def get( self, key ):
        try:
            return self.__map[ key ]
        except:
            return None

    def __str__( self ):
        for [key,value] in self.__map.items():
            print( key, value )
    


trainDatas = Dataset()


class Item: 
    def __init__( self, id, dt, dlg, dlt, at, alg, alt, t, p ):
        self.id = id
        self.dt = self.time( dt )
        self.dlg = dlg
        self.dlt = dlt
        self.at = self.time( at )
        self.alg = alg
        self.alt = alt
        self.dist = self.distance()
        self.t = t
        self.p = p
    

    def time( self, string ):
        # +7200 = +2hours because of utc-time difference
        return int( time.mktime(datetime.datetime.strptime( string, "%Y-%m-%d %H:%M:%S").timetuple()) ) + 7200

    def duration( self ):
        return self.at - self.dt

    def distance( self ):
        R = 6372800  # Earth radius in meters
        lat1, lon1 = float(self.dlt), float(self.dlg)
        lat2, lon2 = float(self.alt), float(self.alg)
        phi1, phi2 = math.radians(lat1), math.radians(lat2) 
        dphi       = math.radians(lat2 - lat1)
        dlambda    = math.radians(lon2 - lon1)
        a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
        return 2*R*math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    def iterable( self ):
        return [self.id,self.dt,self.dlg,self.dlt,self.at,self.alg,self.alt,self.dist,self.t,self.p]

    def __str__(self):
        return "['p':'{0}','pt':'{1}','plg':'{2}','plt':'{3}','dt':'{4}','dlg':'{5}','dlt':'{6}','dist':'{7}','t':'{8}','cp':'{9}']".format( self.id, self.dt, self.dlg, self.dlt, self.at, self.alg, self.alt, self.distance(), self.t, self.p )

def readFile ( string ):
    out = open( 'ai.{}'.format( string ), 'w' )
    out.truncate(0)
    writer = csv.writer( out )
    with open( string, 'r') as f:
        reader = csv.reader(f)
        next( reader )
        for row in reader:
            newItem = Item( row[0], row[2], row[5], row[6], row[3], row[7], row[8], row[10], row[4] )
            trainDatas.set( row[0], newItem )
            writer.writerow( newItem.iterable() )

readFile( 'train.csv' )
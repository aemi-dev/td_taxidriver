import csv

class BoxCoordonnée:
    def __init__(self,min_x_min_y,max_x_min_y,min_x_max_y,max_x_max_y):
        self.min_x_min_y = min_x_min_y
        self.max_x_min_y = max_x_min_y
        self.min_x_max_y = min_x_max_y
        self.max_x_max_y = max_x_max_y

    @classmethod
    def get_min_x_min_y(self):
        return self.min_x_min_y

    def get_max_x_min_y(self):
        return self.max_x_min_y

    def get_min_x_max_y(self):
        return self.min_x_max_y
    
    def get_max_x_max_y(self):
        return self.max_x_max_y


class Coordonnée:
    def __init__(self,longitude,latitude):
        self.longitude = longitude
        self.latitude = latitude

    @classmethod
    def getlongitude(self):
        return self.getlongitude
    
    def getlattitude(self):
        return self.getlattitude

class District:
    def __init__(self,Coord):
        self.Coord = Coord

    @classmethod
    def getDistrict(self):
        #A faire for each row (1) vérifier getlattitude, getlongitude du coupe réfléchir sur l'ordre > ou <
        for i in DistrictNewYork:
            #Il reste à faire chaque coordonnée une à une.
            if (self.Coord.getlongitude < i[1].get_min_x_min_y.getlongitude and
                self.Coord.getlattitude < i[1].get_min_x_min_y.getlattitude):
                return i[0]

#Il reste enormement de quartier à faire
DistrictNewYork = [["Upper East Side"], [BoxCoordonnée(Coordonnée(-73.9731,40.7657),
                                        Coordonnée(-73.9591,40.7572),
                                        Coordonnée(-73.9559,40.7894),
                                        Coordonnée(-73.9397,40.7807))]
    
    
    ]
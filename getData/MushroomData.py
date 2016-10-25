#!/usr/bin/python

#MushroomData.py
#
#SENG474 Project
#Group: Alix Voorthuyzen, 
#       Alice Gibbons, 
#       Jason Curt, 
#       Matthew Clarkson, 
#       Zachary Seselja
#
#Purpose: Get the mushroom data, transform it into a format usable by
#           the data mining algorithms, split into training and test sets,
#           and process results.
#
import numpy as np
import sys
import os

class MushroomData:
    #file name of raw data
    data_file = ''

    #Mappings: raw data -> numeric data
    #Attribute Information: (classes: edible=e, poisonous=p)
    _class_dict = {'e':1, 'p':-1}

    # 1. cap-shape:                bell=b,conical=c,convex=x,flat=f,
    #                              knobbed=k,sunken=s
    _cap_shape_dict = {'b':1,'c':2,'x':3,'f':4,'k':5,'s':6}

    # 2. cap-surface:              fibrous=f,grooves=g,scaly=y,smooth=s
    _cap_surface_dict = {'f':1,'g':2,'y':3,'s':4}
    
    # 3. cap-color:                brown=n,buff=b,cinnamon=c,gray=g,green=r,
    #                              pink=p,purple=u,red=e,white=w,yellow=y
    _cap_color_dict = {'n':1,'b':2,'c':3,'g':4,'r':5,'p':6,'u':7,'e':8,'w':9,'y':10}

    # 4. bruises?:                 bruises=t,no=f
    _bruises_dict = {'t':1,'f':2}

    # 5. odor:                     almond=a,anise=l,creosote=c,fishy=y,foul=f,
    #                              musty=m,none=n,pungent=p,spicy=s
    _odor_dict = {'a':1,'l':2,'c':3,'y':4,'f':5,'m':6,'n':7,'p':8,'s':9}

    # 6. gill-attachment:          attached=a,descending=d,free=f,notched=n
    _gill_attach_dict = {'a':1,'d':2,'f':3,'n':4}

    # 7. gill-spacing:             close=c,crowded=w,distant=d
    _gill_space_dict = {'c':1,'w':2,'d':3}

    # 8. gill-size:                broad=b,narrow=n
    _gill_size_dict = {'b':1,'n':2}

    # 9. gill-color:               black=k,brown=n,buff=b,chocolate=h,gray=g,
    #                              green=r,orange=o,pink=p,purple=u,red=e,
    #                              white=w,yellow=y
    _gill_color_dict = {'k':1,'n':2,'b':3,'h':4,'g':5,'r':6,'o':7,'p':8,'u':9,'e':10,'w':11,'y':12}

    #10. stalk-shape:              enlarging=e,tapering=t
    _stalk_shape_dict = {'e':1,'t':2}

    #11. stalk-root:               bulbous=b,club=c,cup=u,equal=e,
    #                              rhizomorphs=z,rooted=r
    _stalk_root_dict = {'b':1,'c':2,'u':3,'e':4,'z':5,'r':6}

    #12. stalk-surface-above-ring: fibrous=f,scaly=y,silky=k,smooth=s
    #13. stalk-surface-below-ring: fibrous=f,scaly=y,silky=k,smooth=s
    _stalk_surf_ring_dict = {'f':1,'y':2,'k':3,'s':4}


    #14. stalk-color-above-ring:   brown=n,buff=b,cinnamon=c,gray=g,orange=o,
    #                              pink=p,red=e,white=w,yellow=y
    #15. stalk-color-below-ring:   brown=n,buff=b,cinnamon=c,gray=g,orange=o,
    #                              pink=p,red=e,white=w,yellow=y
    _stalk_color_ring_dict = {'n':1,'b':2,'c':3,'g':4,'o':5,'p':6,'e':7,'w':8,'y':9}

    #16. veil-type:                partial=p,universal=u
    _veil_type_dict = {'p':1,'u':2}

    #17. veil-color:               brown=n,orange=o,white=w,yellow=y
    _veil_color_dict = {'n':1,'o':2,'w':3,'y':4}

    #18. ring-number:              none=n,one=o,two=t
    _ring_num_dict = {'n':1,'o':2,'t':3}

    #19. ring-type:                cobwebby=c,evanescent=e,flaring=f,large=l,
    #                              none=n,pendant=p,sheathing=s,zone=z
    _ring_type_dict = {'c':1,'e':2,'f':3,'l':4,'n':5,'p':6,'s':7,'z':8}

    #20. spore-print-color:        black=k,brown=n,buff=b,chocolate=h,green=r,
    #                              orange=o,purple=u,white=w,yellow=y
    _spore_print_color_dict = {'k':1,'n':2,'b':3,'h':4,'r':5,'o':6,'u':7,'w':8,'y':9}

    #21. population:               abundant=a,clustered=c,numerous=n,
    #                              scattered=s,several=v,solitary=y
    _pop_dict = {'a':1,'c':2,'n':3,'s':4,'v':5,'y':6}

    #22. habitat:                  grasses=g,leaves=l,meadows=m,paths=p,
    #                              urban=u,waste=w,woods=d
    _habitat_dict = {'g':1,'l':2,'m':3,'p':4,'u':5,'w':6,'d':7}

    _dict_seq = [_cap_shape_dict,_cap_surface_dict,_cap_color_dict,_bruises_dict,_odor_dict,
                 _gill_attach_dict,_gill_space_dict,_gill_size_dict,_gill_color_dict,_stalk_shape_dict,
                 _stalk_root_dict,_stalk_surf_ring_dict,_stalk_surf_ring_dict,_stalk_color_ring_dict,_stalk_color_ring_dict,
                 _veil_type_dict,_veil_color_dict,_ring_num_dict,_ring_type_dict,_spore_print_color_dict,
                 _pop_dict,_habitat_dict]
    
    def __init__(self, data_file='./MushDataSet.cvs'):
        assert(os.path.exists(data_file))
        self.data_file = data_file

    #Get the datasets from the csv file
    #   returns list of classes y, and attribute matrix X
    def get_datasets(self, eliminate_missing=True):
        #open and read the data file
        y = []
        X = []
        with open(self.data_file) as f:
            for line in f:
                m = line.strip().split(',')
                #first column is the classes: poisonous or edible
                y.append(self._class_dict[m[0]])
                #other columns are the attributes
                if m[11] is not '?':
                    #only attr 11 "stalk-root" can be unknown, handle it separately
                    x_i = [self._dict_seq[i-1][m[i]] for i in range(1,len(m))]
                    X.append(x_i)
                elif eliminate_missing is True:
                    #attribute 11 is missing eliminate this object
                    pass
                else:
                    #pad data with all possible combinations of attr 11
                    for key,val in self._dict_seq[10].iteritems():
                        x_i = [self._dict_seq[i-1][m[i]] 
                                if i is not 11 else val
                                for i in range(1,len(m))]
                        X.append(x_i)
         
        return y,X



if __name__ == "__main__":
    print("testing data class")
    
    data = MushroomData()

    y, X = data.get_datasets()

    print(y[0])
    print(X[0])

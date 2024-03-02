# Copyright (c) 2019 Parallel Domain, Inc.
# All rights reserved.
# 
# Use of this file is only permitted if you have entered into a separate written license agreement with Parallel Domain Inc.

import string
from collections import namedtuple
import json

import numpy as np

import cv2

import os

# a label and all meta information
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'Car', 'Person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images

    'cuboid_id'   , # ID that is used for 3d cuboid annotations

    'is_thing'    , # Whether this label distinguishes between single instances or not

    'color'       , # The color of this label (used for generating human readable images)
] )

labels = [
    #      Name                              id  cuboid_id is_thing          color               
    Label( "Animal",                          0,        -1,    True, (220, 20,180) ), 
    Label( "Bicycle",                         1,         8,    True, (119, 11, 32) ),  
    Label( "Bicyclist",                       2,         0,    True, ( 64, 64, 64) ),
    Label( "Building",                        3,        -1,   False, ( 70, 70, 70) ), 
    Label( "Bus",                             4,         3,    True, (  0, 60,100) ),
    Label( "Car",                             5,         2,    True, (  0,  0,142) ), 
    Label( "Caravan/RV",                      6,         3,    True, (  0,  0, 90) ), 
    Label( "ConstructionVehicle",             7,        -1,    True, ( 32, 32, 32) ), 
    Label( "CrossWalk",                       8,        -1,    True, (255,255,255) ), 
    Label( "Fence",                           9,        -1,   False, (190,153,153) ), 
    Label( "HorizontalPole",                 10,        -1,    True, (153,153,153) ), 
    Label( "LaneMarking",                    11,        -1,   False, (220,220,220) ), 
    Label( "LimitLine",                      12,        -1,   False, (180,180,180) ), 
    Label( "Motorcycle",                     13,         4,    True, (  0,  0,230) ), 
    Label( "Motorcyclist",                   14,        11,    True, (128,128,128) ),
    Label( "OtherDriveableSurface",          15,        -1,   False, ( 80,  0,  0) ), 
    Label( "OtherFixedStructure",            16,        -1,   False, (150,  0,  0) ), 
    Label( "OtherMovable",                   17,        -1,    True, (230,  0,  0) ), 
    Label( "OtherRider",                     18,        -1,    True, (192,192,192) ), 
    Label( "Overpass/Bridge/Tunnel",         19,        -1,   False, (150,100,100) ), 
    Label( "OwnCar(EgoCar)",                 20,         2,   False, (128,230,128) ), 
    Label( "ParkingMeter",                   21,        -1,   False, ( 32, 32, 32) ),
    Label( "Pedestrian",                     22,         0,    True, (220, 20, 60) ), 
    Label( "Railway",                        23,        -1,   False, (230,150,140) ), 
    Label( "Road",                           24,        -1,   False, (128, 64,128) ), 
    Label( "RoadBarriers",                   25,        -1,   False, ( 80, 80, 80) ), 
    Label( "RoadBoundary(Curb)",             26,        -1,   False, (100,100,100) ), 
    Label( "RoadMarking",                    27,        -1,   False, (255,220,  0) ), 
    Label( "SideWalk",                       28,        -1,   False, (244, 35,232) ), 
    Label( "Sky",                            29,        -1,   False, ( 70,130,180) ), 
    Label( "TemporaryConstructionObject",    30,        -1,    True, (255,160, 20) ), 
    Label( "Terrain",                        31,        -1,   False, ( 81,  0, 81) ), 
    Label( "TowedObject",                    32,         9,    True, (  0,  0,110) ), 
    Label( "TrafficLight",                   33,        -1,    True, (250,170, 30) ), 
    Label( "TrafficSign",                    34,        -1,    True, (220,220,  0) ), 
    Label( "Train",                          35,         6,    True, (  0, 80,100) ), 
    Label( "Truck",                          36,         1,    True, (  0,  0, 70) ), 
    Label( "Vegetation",                     37,        -1,   False, (107,142, 35) ),   
    Label( "VerticalPole",                   38,        -1,    True, (153,153,153) ), 
    Label( "WheeledSlow",                    39,         5,    True, (  0, 64, 64) ),
    Label( "LaneMarkingOther",               40,        -1,   False, (255,255,  0) ), 
    Label( "LaneMarkingGap",                 41,        -1,   False, (  0,255,255) ), 

    Label( "Fence(Transparent)",             42,        -1,   False, ( 85, 75, 75) ), 
    Label( "StaticObject(Trashcan)",         43,        -1,   False, ( 75,  0,  0) ), 
    Label( "Vegetation(Bush)",               44,        -1,   False, ( 54, 71, 18) ), 

    Label( "OtherPole",                      45,        -1,   False, (200,200,200) ), 

    Label( "Powerline",                      46,        -1,   False, ( 32, 32, 32) ), 
    
    Label( "SchoolBus",                      47,        -1,    True, ( 15,123,122) ),

    Label( "ParkingLot",                     48,        -1,   False, (104, 27, 83) ),

    Label( "RoadMarkingSpeed",               49,        -1,   False, (228,150, 49) ),

    Label( "Vegetation(GroundCover)",        50,        -1,   False, ( 35, 46, 11) ),
    Label( "Vegetation(Grass)",              51,        -1,   False, ( 47,106, 45) ),
    Label( "Vegetation(Tree)",               52,        -1,   False, (107,142, 35) ),

    Label( "Debris",                         53,        -1,   True, ( 80, 41, 21) ),

    Label( "RoadBoundary(CurbFlat)",         54,        -1,   False, (120,120,120) ),
    
    Label( "LaneMarking(Parking)",           55,        -1,   False, (210,210,210) ), 
    Label( "LaneMarking(ParkingIndicator)",  56,        -1,   False, (210,220,210) ), 
    Label( "RoadMarkingArrows",              57,        -1,   False, (228,190, 60) ), 
    Label( "RoadMarkingBottsDots",           58,        -1,   False, (228,120, 49) ), 
    Label( "StopLine",                       59,        -1,   False, (180,150,150) ),

    Label( "ChannelizingDevice",             60,        -1,    True, (237,190,120) ),

    Label( "LaneMarkingSpan",                61,        -1,   False, (  0,180,255) ),

    Label( "StaticObject(BikeRack)",         62,        -1,    True, ( 75,  0, 75) ),

    Label( "ParkingSpot",                    63,        -1,    True, ( 84,155,205) ),

    Label( "RoadBoundary(CurbTop)",          64,        -1,   False, ( 140,140,140) ),
    Label( "RoadBoundary(CurbSide)",         65,        -1,   False, ( 140,160,140) ),
    Label( "RoadBoundary(CurbRoadLevel)",    66,        -1,   False, ( 140,180,140) ),
    
    Label( "Void",                          255,        -1,   False, (  0,  0,  0) )
]

# Cuboid labels
# 0 - Pedestrian
# 1 - Truck
# 2 - Car
# 3 - Bus/Caravan/RV
# 4 - Motorcycle
# 5 - WheeledSlow
# 6 - Train
# 7 - 
# 8 - Bicycle
# 9 - Trailer
# 10 - Bicyclist
# 11 - Motorcyclist

#--------------------------------------------------------------------------------
# Create dictionaries for a fast lookup
#--------------------------------------------------------------------------------

# name to label object
name_to_label      = { label.name    : label for label in labels }
# id to label object
id_to_label        = { label.id      : label for label in labels }
# id to label color tuple
id_to_color        = { label.id      : label.color for label in labels }

id_to_cuboid_id    = { label.id      : label.cuboid_id for label in labels if label.cuboid_id != -1 }

id_to_is_thing     = { label.id      : label.is_thing for label in labels }

id_to_color_lookup = np.zeros((256, 3), dtype=np.uint8)
for label in labels:
    id_to_color_lookup[label.id] = label.color
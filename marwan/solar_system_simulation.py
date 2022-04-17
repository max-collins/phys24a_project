import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.patches import Rectangle, Circle 
from p24asolver import P24ASolver
import pygame
import math


#In this file we will try to simulate the solar system visually using pygame

#now let's create a class for each of the plannets 
class Planet():
    '''
    This class is used to store the valuable information regarding a plannet,
    particularly it's mass, radius, radius of orbit around the sun, and period
    around the sun.
    
    Assume circular orbits and no initial angular displacement from the horizontal
    
    For the units we will use astronomical units (thus radius between earth and sun)
    is 1AU, and we measure the period in years.
    
    We may use that T^2 = 4pi^2/GM  r^3 (Kepler's) third law in our derivation

    #AU's and solar masses, and years
    '''
    

    def __init__(self, mass, radius, radius_of_orbit, period, image_file = 'jupiter.jpg'):
        '''
        inputs:
            mass
            radius
            radius_of_orbit
            period 
        outputs:
            a class which holds all the information regarding an orbiting plannet
        '''
        
        self.mass = mass
        self.radius = radius
        self.radius_of_orbit = radius_of_orbit
        self.period = period
        #initialize the position of all plannets to be collinear with horizontal
        self.pos = [self.radius_of_orbit*1,self.radius_of_orbit*0]
        #now let's calculate omega...
        self.omega = (2*math.pi)/ ((self.radius)**1.5)
        self.sprite_image = pygame.image.load(image_file)


    def update_position(self, time):
        '''
        Update the position of the plannet with respect to some time value

        note that using astronomical units we have... w = 2pi/T = 2pi/(a^3/2) and thus
        we get our position that we may update...
        '''
        
        self.pos = [
        self.radius_of_orbit*math.cos(self.omega*time), 
        self.radius_of_orbit*math.sin(self.omega*time)
        ]




#Now let's create some planets
jupiter = Planet(0.000954588, 0.0005, 5.2, 11.86)





from solar_system_simulation import *


#create a visual space using pygame
pygame.init()
clock = pygame.time.Clock()

#now we create the screen
screen_size = (800,600)
screen = pygame.display.set_mode(screen_size)
caption = "Gravitational Slingshot Animation"
pygame.display.set_caption(caption)

#create the background which we will draw everything to
background = pygame.Surface(screen_size)
#set the background_color 
background_color = (0,0,0)

#deltat/other constants
t = 0.1
delta_t = 0.1
planet_list = [jupiter]

#Make the simulation active
simulation_active = True

while simulation_active:
	for event in pygame.event.get():
		#handles what happens when we click the exist button
		if event.type == pygame.QUIT:
			#end the simulation
			simulation_active = False

	#Now we update the screen and the background
	background.fill(background_color)

	#now we update position
	#now we draw
	for planet in planet_list:
		planet.update_position(t)
		background.blit(planet.sprite_image, planet.pos)


	#update the current time
	t = t + delta_t

	#draw the background onto the screen
	screen.blit(background, (0,0))
	pygame.display.flip()
	clock.tick(60)
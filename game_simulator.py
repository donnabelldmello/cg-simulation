from scipy.spatial import distance
from decimal import *
import numpy as np
from game_utility import *


################ CONFIGURED INPUT ##############
no_of_ferries = 2
no_of_discrete_time_intervals = 16
maximam_velocity_vector = [1, 2]
port_coordinates_vector = [[0,0],[0,8]]
no_of_trips_vector = [2,2]
halt_time_at_port = 0
buffer_before_start = 0

render = False
p_radius = 0.2
no_of_patrollers = 1
patroller_probability_vector = [0.7]

showLegend = False
##############  CALCULATED INPUT  ##############
pSchedule = np.array([[0.3, 0.5, 0.8, 0.7, 0.6, 0.5, 0.4, 0.4, 0.2, 0.3, 0.4, 0.5, 0.5, 0.5, 0.6, 0.8]])
target = [0, 0.4]
################################################



class game_simulator(object):

	def __init__(self, no_of_ferries, no_of_discrete_time_intervals, maximam_velocity_vector, port_coordinates_vector, no_of_trips_vector, 
			halt_time_at_port, buffer_before_start, render, p_radius, no_of_patrollers, patroller_probability_vector):
		super(game_simulator, self).__init__()
		self.no_of_ferries = no_of_ferries
		self.no_of_discrete_time_intervals = no_of_discrete_time_intervals
		self.maximam_velocity_vector = maximam_velocity_vector
		self.port_coordinates_vector = port_coordinates_vector
		self.no_of_trips_vector = no_of_trips_vector
		self.halt_time_at_port = halt_time_at_port
		self.buffer_before_start = buffer_before_start
		self.render = render
		self.p_radius = p_radius
		self.no_of_patrollers = no_of_patrollers
		self.patroller_probability_vector = patroller_probability_vector


	def getLinearSchedule(self, schedule, timeStep, startTime, dst):
		vmax = self.maximam_velocity_vector
		trips = self.no_of_trips_vector

		for fIndex,fItem in enumerate(schedule):
				for tIndex, tItem in enumerate(schedule[fIndex]):
					position = (vmax[fIndex] * timeStep * (tIndex)) - startTime[fIndex]
					rangeStart = game_utility.findRangeStart(position, dst)
					if((rangeStart/dst) > trips[fIndex] + 1):
						position = 0
					else:
						if(position > dst and (rangeStart/dst)%2 != 0):
							# RETURNING FERRY
							position = dst - (position - rangeStart)
						
						elif (position > dst and (rangeStart/dst)%2 == 0):
							# MOVING FORWARD FERRY
							position = position - rangeStart;
					schedule[fIndex][tIndex] = format(max(game_utility.normalize(position, rangeStart+1, dst), 0.0), '.2f')
					#schedule[fIndex][tIndex] = format(max(position, 0), '.2f')

		return schedule
	
	# method: getFerrySchedule
	# description: This method generates the ferry schedule, depending on certain parameters.
	# params :-
		# f : No of ferries
		# t : discrete time intervals (in hrs)
		# vmax : array of maximam speed possible for a respective ferry (km/hr)
		# ports : 2D array of x-y co-ordinates of ports (km)
		# trips : array of trips to be made
		# haltTime : Time to halt at the port
		# startBuffer : buffer in start time (to be use for each ferry)

	def getFerrySchedule(self):
		f = self.no_of_ferries
		t = self.no_of_discrete_time_intervals
		vmax = self.maximam_velocity_vector
		ports = self.port_coordinates_vector
		trips = self.no_of_trips_vector
		haltTime = self.halt_time_at_port
		startBuffer = self.buffer_before_start

		schedule = np.array([[0.0 for x in range(t)] for y in range(f)])
		
		#Find distance from port co-ordinates
		portA = ports[0]
		portB = ports[1]
		self.dst = dst = distance.euclidean(portA, portB)

		finishTime = [0 for x in range(f)]
		startTime = [0 for x in range(f)]

		for fIndex in range(f):
			if(fIndex > 0):
				startTime[fIndex] = startTime[fIndex - 1] + startBuffer #TODO: Randomize start time
			tripTime = ((2 * dst * trips[fIndex])/vmax[fIndex])  + haltTime
			finishTime[fIndex] = (startTime[fIndex] + tripTime)

		timeStep = max(finishTime)/t;
		print("Time step: %f hrs" % timeStep)
		print("Total time: %s hrs" % format(max(finishTime), '.2f'))

		schedule = self.getLinearSchedule(schedule, timeStep, startTime, dst)		
		return schedule;

	def runGame(self, fSchedule, pSchedule, target):
		activePatrollers = []
		render = self.render
		unsuccessfulProbability = 0
		attackTimeStmp = game_utility.denormalize(target[1], self.dst)
		targetPosition = fSchedule[target[0]][game_utility.denormalize(target[1], self.dst)]
		for pIndex, pItem in enumerate(pSchedule):
			if (abs(pSchedule[pIndex][attackTimeStmp] - targetPosition) <= self.p_radius):
				print("Patroller position: %f" %  pSchedule[pIndex][attackTimeStmp])
				print("Target position: %f" % targetPosition)
				unsuccessfulProbability = unsuccessfulProbability + self.patroller_probability_vector[pIndex]
				activePatrollers.append(pIndex)

		if(unsuccessfulProbability > 0):
			return (+1, -1), unsuccessfulProbability, activePatrollers
		else:
			return (-1, +1)

print("No of ferries (f): %d" % no_of_ferries)
print("Discrete time intervals (t): %d" % no_of_discrete_time_intervals)
print("No of trips:")
print(','.join(['{:4}'.format(item) for item in no_of_trips_vector]))
print("Velocity (km/hr):")
print(','.join(['{:4}'.format(item) for item in maximam_velocity_vector]))
print("\n--------------  Simulation Results  ----------------")
simulator = game_simulator(no_of_ferries,no_of_discrete_time_intervals,maximam_velocity_vector, port_coordinates_vector,no_of_trips_vector, 
	halt_time_at_port,buffer_before_start, render, p_radius, no_of_patrollers, patroller_probability_vector)

fSchedule = simulator.getFerrySchedule()
print("Ferry schedule: ")
print('\n'.join(['\t'.join(['{:4}'.format(item) for item in row]) for row in fSchedule]))

print("\nTarget: ")
print(','.join(['{:4}'.format(item) for item in target]))

reward, prob, activePatrollers = simulator.runGame(fSchedule, pSchedule, target)
print("Reward: %s" % format(reward))
print("Unsuccessful probability: %s" % format(prob))
print("Active Patrollers: %s" % format(activePatrollers))

game_utility.renderSimulation(8, no_of_discrete_time_intervals, fSchedule, pSchedule, target, simulator.dst, activePatrollers, showLegend)

	
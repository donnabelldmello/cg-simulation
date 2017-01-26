import numpy as np
import logging, sys
from decimal import *
from game_utility import *
import matplotlib.pyplot as plt
from scipy.spatial import distance

class game_simulator():

	def __init__(self, no_of_ferries, no_of_discrete_time_intervals, maximam_velocity_vector, port_coordinates_vector, no_of_trips_vector, 
			halt_time_at_port, buffer_before_start, render, show_legend, p_radius, no_of_patrollers, patroller_probability_vector, verbose, pSchedule):
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
		self.show_legend = show_legend
		self.patroller_probability_vector = patroller_probability_vector
		self.pSchedule = pSchedule
		if(verbose):
			logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
		else:
			logging.basicConfig(stream=sys.stderr, level=logging.ERROR)


	def getLinearSchedule(self, schedule, startTime):
		"""
		This method gets ferry scheule for the linear(basic version)
		Returns schedule (position of ferry) in an f x n array. Where f = no of ferries and n = discrete time intervals
		"""
		time_step = self.time_step
		vmax = self.maximam_velocity_vector
		trips = self.no_of_trips_vector
		dst = self.dst

		for fIndex,fItem in enumerate(schedule):
			forwardDirection = True
			tripNo = 1
			for tIndex, tItem in enumerate(fItem):
				position = (vmax[fIndex] * ((time_step * tIndex) + startTime[fIndex]))
				rangeStart = game_utility.findRangeStart(position, dst)
				if(position > dst and (rangeStart/dst)%2 != 0):
					# RETURNING FERRY
					position = dst - (position - rangeStart)
					if(forwardDirection):
						#tripNo = tripNo + 1
						forwardDirection = False
						#print("return", position)
				elif (position > dst and (rangeStart/dst)%2 == 0):
					# MOVING FORWARD FERRY
					position = position - rangeStart;
					if(not forwardDirection):
						tripNo = tripNo + 1
						forwardDirection = True
						#print("forward", position)
				#print(format(max(game_utility.normalize(position, dst), 0.0), '.2f'))
				#print(rangeStart)
				if(tripNo > trips[fIndex]):
					position = 0
				schedule[fIndex][tIndex] = format(max(game_utility.normalize(position, dst), 0.0), '.2f')
		return schedule

	def reset(self):
		"""
		Generates the ferry schedule.
		Returns schedule (position of ferry) in an f x n array. Where f = no of ferries and n = discrete time intervals
		"""

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

		finishTime = [0.0 for x in range(f)]
		startTime = [0.0 for x in range(f)]

		#Calculate total time for all ferries to complete required trips considering respective maximum velocities
		for fIndex in range(f):
			if(fIndex > 0):
				startTime[fIndex] = startTime[fIndex - 1] + startBuffer #TODO: Randomize start time
			tripTime = ((2 * dst * trips[fIndex])/vmax[fIndex])  + haltTime
			finishTime[fIndex] = (startTime[fIndex] + tripTime)

		self.time_step = time_step = max(finishTime)/(t-1);
		logging.debug("Time step: %f hrs" % time_step)
		logging.debug("Total time: %s hrs" % format(max(finishTime), '.2f'))

		self.fSchedule = schedule = self.getLinearSchedule(schedule, startTime)	
		return schedule;

	def runGame(self, target):
		"""
		Simulates attack and calculates rewards(zero sum) for attacker and defender.
		If patroller is within a radius of target ferry, attack is unsuccessful with probability else attack is successful
	
		Returns (+1, -1), unsuccessfulProbability for unsuccessful attack
		else return (-1, +1), 0 if attack is successful
		"""
		fSchedule = self.fSchedule
		pSchedule = self.pSchedule
		
		activePatrollers = []
		unsuccessfulProbability = 0
		attackTimeStmp = game_utility.denormalize(target[1], self.no_of_discrete_time_intervals-1)
		attackTime = int(attackTimeStmp)
		targetX = target[1]
		targetY = game_utility.getNormalizedPosition(target[0], fSchedule, attackTimeStmp, target[1])
		targetPosition = fSchedule[target[0]][attackTime]
		
		for pIndex, pItem in enumerate(pSchedule):
			patrollerX = target[1]
			patrollerY = game_utility.getNormalizedPosition(pIndex, pSchedule, attackTimeStmp, target[1])

			dist_from_attack = distance.euclidean([targetX, targetY], [patrollerX, patrollerY])
			logging.debug("Patroller ferry dist: %f" % dist_from_attack)
			logging.debug("Radius: %f" % p_radius)
			logging.debug("Patroller position: %f, %f" %  (patrollerX, patrollerY))
			logging.debug("Target position: %f , %f" % (targetX, targetY))

			if (dist_from_attack <= self.p_radius):
				unsuccessfulProbability = unsuccessfulProbability + self.patroller_probability_vector[pIndex]
				activePatrollers.append(pIndex)
				logging.debug("ATTACK FAILED")

		if(self.render):
			self.renderSimulation(target, activePatrollers, attackTimeStmp, targetPosition)

		logging.debug("Active Patrollers: %s" % format(activePatrollers))
		logging.debug("Unsuccessful probability: %s" % format(unsuccessfulProbability))
		if(unsuccessfulProbability > 0):
			return (+1, -1)
		else:
			return (-1, +1)

	def renderSimulation(self, target, activePatrollers, attackTime, targetPosition):
		"""
		Plots paths for ferry and patrollers & highlights attacked position and active patroller 

		"""

		t = self.no_of_discrete_time_intervals
		fSchedule = self.fSchedule
		pSchedule = self.pSchedule
		show_legend = self.show_legend

		xaxis = np.array([1.0*(x)/(t-1) for x in range(t)])
		attack_ferry = target[0]
		legendArr = []

		# Attack on target plot
		attackPosition = game_utility.getNormalizedPosition(target[0], fSchedule, attackTime, target[1])
		plt.plot([target[1]], [attackPosition], 'ro')
		legendArr.append("Attack")

		# Attacked ferry plot
		plt.plot(xaxis, fSchedule[attack_ferry], '--')
		legendArr.append("Ferry" + format(attack_ferry))

		#Other ferry plots
		for f in range(len(fSchedule)):
			if(f != attack_ferry):
				plt.plot(xaxis, fSchedule[f], '--')
				legendArr.append("Ferry" + format(f))

		#Patroller plots
		for p in range(len(pSchedule)):
			plt.plot(xaxis, pSchedule[p], '--')
			legendArr.append("Patroller" + format(p))

		# Active Patroller
		for activeP in range(len(activePatrollers)):
			attackPosition = game_utility.getNormalizedPosition(activeP, pSchedule, attackTime, target[1])
			plt.plot([target[1]], [attackPosition], 'gs')
			legendArr.append("Active patrollers")

		if(show_legend):
			plt.legend(legendArr, loc="upper right")

		plt.axis([0, 1, 0, 1])
		#plt.xticks(np.arange(min(xaxis), max(xaxis), 0.04))    # adjust number of ticks
		#plt.grid()
		plt.xlabel('Time')
		plt.ylabel('Distance')
		plt.show()

if __name__ == "__main__":

	################ CONFIGURED INPUT ##############
	no_of_ferries = 2
	no_of_discrete_time_intervals = 35
	maximam_velocity_vector = [1, 1]
	port_coordinates_vector = [[0,0],[0,8]]
	no_of_trips_vector = [3,2]
	halt_time_at_port = 0
	buffer_before_start = 0

	p_radius = 0.21
	no_of_patrollers = 1
	patroller_probability_vector = [0.7]

	render = True
	show_legend = False
	verbose = True
	pSchedule = np.array([[0.0, 0.5, 0.6, 0.7, 1.0, 0.0, 0.5, 0.6, 0.7, 1.0, 0.0, 0.5, 0.6, 0.7, 1.0, 0.0, 0.5, 0.6, 0.7, 1.0, 0.0, 0.5, 0.6, 0.7, 1.0,  0.0, 0.5, 0.6, 0.7, 1.0, 0.0, 0.5, 0.6, 0.7, 1.0]])
	##############  CALCULATED INPUT  ##############
	target = [0, 0.63]
	################################################

	simulator = game_simulator(no_of_ferries,no_of_discrete_time_intervals,maximam_velocity_vector, port_coordinates_vector,no_of_trips_vector, 
	halt_time_at_port,buffer_before_start, render,show_legend, p_radius, no_of_patrollers, patroller_probability_vector, verbose, pSchedule)

	logging.debug("\n---------------------  Setup  ---------------------")
	logging.debug("No of ferries (f): %d" % no_of_ferries)
	logging.debug("Discrete time intervals (t): %d" % no_of_discrete_time_intervals)
	logging.debug("No of trips:")
	logging.debug(','.join(['{:4}'.format(item) for item in no_of_trips_vector]))
	logging.debug("Velocity (km/hr):")
	logging.debug(','.join(['{:4}'.format(item) for item in maximam_velocity_vector]))

	logging.debug("\n--------------  Simulation Results  ----------------")

	fSchedule = simulator.reset()
	logging.debug("Ferry schedule: ")
	logging.debug('\n'.join(['\t'.join(['{:4}'.format(item) for item in row]) for row in fSchedule]))

	logging.debug("Patroller schedule: ")
	logging.debug('\n'.join(['\t'.join(['{:4}'.format(item) for item in row]) for row in pSchedule]))

	logging.debug("\nTarget: ")
	logging.debug(','.join(['{:4}'.format(item) for item in target]))

	reward = simulator.runGame(target)
	logging.debug("Reward: %s" % format(reward))
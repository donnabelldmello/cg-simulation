import matplotlib.pyplot as plt
import numpy as np

class game_utility():
	def __init__(self):
        	pass

	@staticmethod
	def findRangeStart(position, dst):
		distance = dst
		while (position > distance):
			distance = distance + dst
		return distance - dst;

	@staticmethod
	def normalize(data, length):
		return data/length;

	@staticmethod
	def denormalize(data, length):
		return int(data * length);

	@staticmethod
	def renderSimulation(d, t, fSchedule, pSchedule, target, dst, activePatrollers, showLegend, attackTime):
		attackTime = attackTime - 1#for indexing subtract 1
		xaxis = np.array([1.0*(x)/(t-1) for x in range(t)])
		attack_ferry = target[0]
		legendArr = []

		#Attack & attacked ferry plot
		plt.plot([target[1]], [fSchedule[attack_ferry][attackTime]], 'ro')
		legendArr.append("Attack")

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

		for activeP in range(len(activePatrollers)):
			plt.plot([target[1]], [pSchedule[activeP][attackTime]], 'gs')
			legendArr.append("Active patrollers")

		if(showLegend):
			plt.legend(legendArr, loc="upper right")

		plt.axis([0, 1, 0, 1])
		plt.xlabel('Time')
		plt.ylabel('Distance')
		plt.show()
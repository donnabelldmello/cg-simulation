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
		return data * length;

	@staticmethod
	def calculateLocation(fIndex, fSchedule, attackTime, targetAttackTime):
		pfirst = int(attackTime)
		pfirst = game_utility.normalize(pfirst*1.0, len(fSchedule[0])-1)
		first = fSchedule[fIndex][int(attackTime)]
		offset = attackTime - int(attackTime) # 17.5 - 17.0 = 0.5
		attackPosition = first

		#print("first", first)
		#print("offset", offset)
		#print("attackPosition", attackPosition)

		if (int(attackTime) < len(fSchedule[0]) - 1):
			psecond = int(attackTime) + 1
			psecond = game_utility.normalize(psecond*1.0, len(fSchedule[0])-1)

			second = fSchedule[fIndex][int(attackTime) + 1]
			diff = second - first # 1.0 - 0.83 = 0.17

			attackPosition = attackPosition + (diff * offset)
			#print("second", second)
			#print("diff", diff)

			#print("psecond", psecond)	
			#print("pfirst", pfirst)

			attackPosition = ((second - first) * (psecond - targetAttackTime))/(psecond - pfirst)
			#print("attackPosition", attackPosition)

			if(attackPosition < 0):
				attackPosition = first - (((first - second) * (pfirst - targetAttackTime))/(pfirst - psecond))
			else:
				attackPosition = second - attackPosition
			#print("attackPosition", attackPosition)
			return attackPosition

	@staticmethod
	def renderSimulation(d, t, fSchedule, pSchedule, target, dst, activePatrollers, showLegend, attackTime, targetPosition):
		attackTime = attackTime#for indexing subtract 1

		xaxis = np.array([1.0*(x)/(t-1) for x in range(t)])
		attack_ferry = target[0]
		legendArr = []

		attackPosition = game_utility.calculateLocation(target[0], fSchedule, attackTime, target[1])
		plt.plot([target[1]], [attackPosition], 'ro')
		#plt.plot([target[1]], [fSchedule[attack_ferry][attackTime]], 'ro')
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
			attackPosition = game_utility.calculateLocation(activeP, pSchedule, attackTime, target[1])
			plt.plot([target[1]], [attackPosition], 'gs')
			legendArr.append("Active patrollers")

		if(showLegend):
			plt.legend(legendArr, loc="upper right")

		plt.axis([0, 1, 0, 1])
		#plt.xticks(np.arange(min(xaxis), max(xaxis), 0.04))
		plt.grid()
		plt.xlabel('Time')
		plt.ylabel('Distance')
		plt.show()
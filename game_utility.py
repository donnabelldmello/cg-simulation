class game_utility():

	@staticmethod
	def findRangeStart(position, dst):
		"""
		Gets start position of a range distance
		E.g. Position 10 with max distance 8, would return startRange as 8 (8-16)
		"""
		distance = dst
		while (position > distance):
			distance = distance + dst
		return distance - dst;

	@staticmethod
	def normalize(data, length):
		"""
		Normalizes data between 1 - length to 0 - 1
		"""
		return data/length;

	@staticmethod
	def denormalize(data, length):
		"""
		De-normalizes data between 0 - 1 to 1 - length
		"""
		return data * length;

	@staticmethod
	def getNormalizedPosition(fIndex, fSchedule, attackTime, targetAttackTime):
		"""
		Calculates location/position(normlized) based on timeStamp(not-normalized)
		"""
		pfirst = int(attackTime)
		pfirst = game_utility.normalize(pfirst*1.0, len(fSchedule[0])-1)
		first = fSchedule[fIndex][int(attackTime)]
		attackPosition = first

		if (int(attackTime) < len(fSchedule[0]) - 1):
			psecond = int(attackTime) + 1
			psecond = game_utility.normalize(psecond*1.0, len(fSchedule[0])-1)
			second = fSchedule[fIndex][int(attackTime) + 1]

			attackPosition = ((second - first) * (psecond - targetAttackTime))/(psecond - pfirst)

			if(attackPosition < 0):
				attackPosition = first - (((first - second) * (pfirst - targetAttackTime))/(pfirst - psecond))
			else:
				attackPosition = second - attackPosition
		return attackPosition
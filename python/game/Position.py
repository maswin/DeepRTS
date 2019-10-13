class Position:
	def __init__(self):
		self.x = -1
		self.y = -1

	def __init__(self, x, y):
		self.x = x
		self.y = y

	def check_out_of_bounds(self, width, height):
		if(self.x < 0 or self.y < 0 or self.x >= width or self.y >= height):
			return False
		else:
			return True
	
	def is_equals(self, p):
		if(self.x == p.x and self.y == p.y):
			return True
		else: 
			return False
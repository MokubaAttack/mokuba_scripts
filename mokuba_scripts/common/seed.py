import random

def make_seed(seed,pic_number):
	if isinstance(seed, list):
		pic_number=len(seed)
		for i in range(pic_number):
			try:
				if int(seed[i])==0:
					seed[i]=random.randint(1, 2**31-1)
				else:
					seed[i]=int(seed[i])
			except:
				seed[i]=random.randint(1, 2**31-1)
	else:
		try:
			if int(seed)==0:
				seed=[]
				for i in range(pic_number):
					seed.append(random.randint(1, 2**31-1))
			else:
				seed=[int(seed)]
				pic_number=1
		except:
			seed=[]
			for i in range(pic_number):
				seed.append(random.randint(1, 2**31-1))
	return seed,pic_number
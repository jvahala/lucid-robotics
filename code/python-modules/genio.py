'''
file: genio.py
purpose: general file i/o functions
'''
def getlines(filename): 
	'''
	input filename ('example.txt') and get dictionary of lines indexed by line number (starting at line 0), and number of lines total
	'''
	lines = {}		#holds lines
	with open(filename,'r') as f:
		c = -1 
		for c,l in enumerate(f): 
			lines[c] = l 
	c += 1
	return lines,c

def appendline(filename,line):
	append = '\n'+line
	with open(filename,'a') as f: 
		f.write(append)

def linecount(filename):
	i = -1
	with open(filename,'r') as f: 
		for i,l in enumerate(f): 
			pass
	return i+1

def shortenfile(filename,lines,numlines): 
	with open(filename,'w') as f: 
		for i in range(numlines):
			f.write(lines[i])
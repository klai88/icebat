# Used the following post as a guideline for this program
#http://forum.codecall.net/topic/50892-tic-tac-toe-in-python/

def print_board():
	for i in range(0,3):
		for j in range(0,3):
			print map[i][j],
			if j != 2:
				print "|",
		print 

def check_done():
	for i in range(0,3):
		for j in range(0,3):
			#row matches, then colm matches
			if map[i][0] == map[i][1] == map[i][2] != " "\
			or map[0][i] == map[1][i] == map[2][i] != " ":
				print turn, "won!!"
				print_board()
				return True
	if map[0][0] == map[1][1] == map[2][2] != " " \
	or map[0][2] == map[1][1] == map[2][0] != " ":
		print turn, "won!"
		print_board()
		return True
	if " " not in map[0] and " " not in map[1] and " " not in map[2]:
		print "Draw"
		return True
		print_board()
	return False




turn = "X"
map = [[" "," "," "],
       [" "," "," "],
       [" "," "," "]]
done = False


while done != True:
    print_board()
    
    print turn+"'s turn"
    print

    moved = False
    while moved != True:
    	print "Select a number to place your move in"
    	print "1 | 2 | 3"
    	print "4 | 5 | 6"
    	print "7 | 8 | 9"
    	print 

    	try:
    		position = input("Select: ")
    		if position <= 9 and position >= 1:
    			row = position/3
    			col = position%3
    			if col == 0:
    				row -=1
    				col = 2
    			else:
    				col -=1
    			if map[row][col] == " ":
    				map[row][col] = turn
    				moved = True
    				done = check_done()

    				if done == False:
    					if turn == "X":
    						turn = "O"
    					else:
    						turn = "X"
    			else:
    				print "Space has already been taken dummy!"
    	except:
    		print "Please enter a number between 1 and 9"
    		print "1 | 2 | 3"
    		print "4 | 5 | 6"
    		print "7 | 8 | 9"


from m2interface import solve

perceptionData = '../data/example.txt'

with open(perceptionData) as f:
    lines = f.readlines()
    
    data = []
    for line_no in range(len(lines)):
        aggr = ""
        for i in range(4, len(lines[line_no].split(' '))):
            aggr += (lines[line_no].split(' ')[i])
        
        data.append(eval(aggr))
        if(line_no%16==15):
            solution = solve(data)
            answer = lines[line_no].split(' ')[3]
           
            if str(answer) in solution:
    	        print(lines[line_no].split(' ')[0], lines[line_no].split(' ')[1], 1/len(solution))
            else:
                print(lines[line_no].split(' ')[0], lines[line_no].split(' ')[1], 0)
            data = [] 

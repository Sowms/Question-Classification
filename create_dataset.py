from nltk import word_tokenize

with open('lab_1.txt') as f:
    content = f.readlines()
f.close()
	
f1 = open('questions.txt', "w")
f2 = open('categories.txt', "w")

for line in content:
	words = line.split(",");
	ans = "unknown"
	if (len(words) != 1):
		ans = words[1]
	if (not ans.strip()):
		ans = "unknown"
	question = words[0].split(" ")[0] + " " + words[0].split(" ")[1] 
	f1.write(question.strip() + "\n")
	f2.write(ans.strip() + "\n")
	
f1.close()
f2.close()
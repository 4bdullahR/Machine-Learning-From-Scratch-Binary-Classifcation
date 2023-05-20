HOW TO RUN THE CODE:
after drag source code file into an IDE you must provide a 
1-CSV file named "diabetes.csv" that contains enough of instances that has six feature and the 7th is the class label.
In our example it is: Glucose,BloodPressure,SkinThickness,Insulin,BMI,Age,Outcome.
	-if your file name is not "diabetes.csv" please follow these steps:
		1-go to line 102 and change the name.
	-if your file has more/less than 7 columns please follow these steps:
		1-in line 18 change the "7" to your number of columns or just delete the hole "len(row)!=7" which it checks an instance
		has 7 feature otherwise, drop it. -MISSING FEATURE IS ANOTHER TOPIC BUT THIS IS JUST IN CASE, THE GIVEN CSV IS FINE-

HOW THE CODE WORKS TO GIVE THE RESULTS:
1-Load the data from the CSV file (get all inctances).
2-Converting the given in string data to float to let our algorithm -logistic regression- handle them.
3-Detrimaining the minimim and the maximum of each feature to normlize them later
4-Since our features in deffirence ranges, we need to normlize the data to a universe scalling.
5-spliting the data into A)traing_set B)test_set.
	NOTE:spliting in everytime is random but, THERE IS A CROSS-VALIDATION FUNCTION that suits our CSV example that will 
	devide the data into three thirds and then train & test 3 times. -not required but I did it-
6-Training the algorithm stars with given data
7-Test the model and see the classifier evaluation using cross-validation, accuracy, precision, recall and F1-score.
all the above 7 is in three time loop to randomly select data to test/train. -you can use Cross Validation function- for none random selction-
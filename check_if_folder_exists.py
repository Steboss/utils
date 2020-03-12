#This is super simple, but I am always forgetting it!!!
#Check if a folder exists otherwise create it - directory as well 
if not os.path.exists(directory):
    os.makedirs(directory)

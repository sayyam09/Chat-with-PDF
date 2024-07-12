
# SETUP_ENVIRONMENT - 
    a) At First, you need to setup an virtual enivornment for you application so setup it first using following command- 

        conda create -p venv python == 3.11.8
    
    b) Activate these environment - 

        conda activate venv/ 

# REQUIREMENTS - 

    a) They are some library/modules/pacakages needed to run our application, so all needed liararies are mentioned 
       into requirements.txt file. Lets install all libararies using following command.

        pip install -r requirements.txt

        
# API_KEY 
    Here, i used API KEY of Google Gemini pro so, create api key from Google AI studio
    and copy paste it into .env file

# DATABASES - 
    1) Connectivity to MySQL:
        In .env file please enter the values for  (DB_HOST, DB_USER, DB_PASSWORD, DB_DATABASE)

    2) Now, go to MySQL :
        
        a) Create an database using following query :

            CREATE db ai_planet_db; // first create an database

        b) Now create an table to store meta data information of uploaded files: 
            
            use ai_planet_db; // select the db in which we want to create the table

            CREATE TABLE files (
                id INT AUTO_INCREMENT PRIMARY KEY,
                filename VARCHAR(255) NOT NULL,
                upload_date DATETIME NOT NULL
            );

#SCREENSHOTS/ DEMO- 
    The screenshots folder contains the screenshots of application how application is working for demo purpose.
    
# RUN - 
    for running the application just run the app.py file.



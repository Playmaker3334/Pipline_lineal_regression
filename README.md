# Pipline_lineal_regression
This is a project in which we will perform a multiple linear regression which will be extracted and mounted in a docker and will be displayed on a web page.

# Points to consider
To ensure proper utilization of the code, the following prerequisites must be met:

    - Account creation at data.world is required to access full code functionality.
    - Installation of libraries listed in the requirements.txt file is essential.
    - Possession of the pertinent CSV files is necessary for data processing.
    - You must to add your own token from data.world.

These steps are crucial for setting up the appropriate working environment and enable effective operation of the provided code.
Step by step guide:
1. Download the github repository (since you're reading this, we can assume you already did this)
2. Activate Docker on your system
    For windows: just open Docker Desktop
    For Linux: Sudo systemctl start docker
3. Open the Dockerfile downloaded from the repository and the next link https://data.world/settings/advanced
4. Create a data.world account or login if you already have one, then copy the API token from the Read/Write section
5. Paste the token in the "your token here" section of the dockerfile
6. Open your console and go to the directory where all documents were downloaded
7. Run the next script on the console #docker build -t mi-proyecto-flask .# (without the hashes) 
8. When the image's creation is done, run the next script on the console #docker run -p 5000:5000 mi-proyecto-flask# (without the hashes)
9. A link will appear on your console, click it to open the app
10. Load the csv documents on the app (the order matters, incd.csv goes in the upper button and death.csv on the other one)
11. Click upload and then training


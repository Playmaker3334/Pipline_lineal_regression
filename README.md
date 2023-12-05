# Pipeline Linear Regression

This project involves performing a multiple linear regression which will be extracted and mounted in a Docker container, and then displayed on a web page.

> :warning: **WARNING**: To ensure proper utilization of the code, the following prerequisites must be met:

- Account creation at data.world is required to access the full functionality of the code.
- Installation of libraries listed in the `requirements.txt` file is essential.
- Possession of the pertinent CSV files is necessary for data processing.
- You must add your own token from data.world.

These steps are crucial for setting up the appropriate working environment and enabling effective operation of the provided code.

## Step by Step Guide:

1. Download the GitHub repository (since you're reading this, we can assume you already did this).
2. Activate Docker on your system:
   - For Windows: just open Docker Desktop.
   - For Linux: `sudo systemctl start docker`.

> :bulb: **TIP**: Visit the following link for advanced settings information on data.world: [https://data.world/settings/advanced](https://data.world/settings/advanced)

3. Create an account on data.world or log in if you already have one, then copy the API token from the Read/Write section.
4. Paste the token into the "your token here" section of the Dockerfile.
5. Open your console and navigate to the directory where all documents were downloaded.
6. Run the following script in the console `docker build -t my-project-flask .` (without the quotes).
7. Once the image creation is complete, run the following script in the console `docker run -p 5000:5000 my-project-flask` (without the quotes).
8. A link will appear in your console, click it to open the app.
9. Load the CSV documents on the app (the order matters, `incd.csv` goes in the upper button and `death.csv` on the other one).
10. Click upload and then training.

> :notebook: **NOTE**: Remember to check and respect the order of the files when uploading to ensure the correct functioning of the application.



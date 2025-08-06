# Graduation Project
This repository contains the code and pipeline for our Graduation Project in speaker verification.

## How to Run
### Prerequisites

- [Docker](https://www.docker.com/) must be installed and running on your machine.
- You must have an account on [Weights & Biases (wandb)](https://wandb.ai/) in order to monitor the experiment runs.
- Create a `.env` file in the root directory of the project with the following variable: WANDB_API_KEY=your_wandb_api_key_here

### Run Instructions
1.  Open Docker Desktop

2.  Open a terminal

3.  Navigate to the project folder:
cd /path/to/graduation-project

4.  Build the Docker image (may take up to ~20 minutes):
docker build -t my-python-app .

5.  Run the Docker container:

Command structure:
docker run --env-file C:\Users\USER\path\to\graduation-project\.env `
  -v C:\Users\USER\path\to\graduation-project:/app `
  my-python-app `
  python main.py "--pipeline" "PipeLineName"
  
* Replace PipeLineName with one of the available options listed below.

#### Example of command
docker run --env-file C:\Users\USER\source\repos\vcuxvdsuk\graduation-project\.env `
  -v C:\Users\USER\source\repos\vcuxvdsuk\graduation-project:/app `
  my-python-app `
  python main.py "--pipeline" "triplet_unsupervised_family"

#### Available --pipeline Options
- triplet_supervised
- triplet_unsupervised
- triplet_unsupervised_family
- ecappa_supervised
- ecappa_unsupervised
- ecappa_unsupervised_family

  

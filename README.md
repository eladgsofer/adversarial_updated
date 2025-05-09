# Adversarial Sensitivity

Welcome to the official repository for the research paper
**"Unveiling and Mitigating Adversarial Vulnerabilities in Iterative Optimizers"**
by Elad Sofer, Tomer Shaked, Caroline Chaux, and Nir Shlezinger.

## Authors

* **Elad Sofer**, **Tomer Shaked**, and **Nir Shlezinger**
  School of Electrical and Computer Engineering, Ben-Gurion University of the Negev, Israel
  Contact: [elad.g.sofer@gmail.com](mailto:elad.g.sofer@gmail.com) | [tosha@post.bgu.ac.il](mailto:tosha@post.bgu.ac.il) | [nirshl@bgu.ac.il](mailto:nirshl@bgu.ac.il)

* **Caroline Chaux**
  CNRS, IPAL, Singapore
  Contact: [caroline.chaux@cnrs.fr](mailto:caroline.chaux@cnrs.fr)

## Installation
To get started, please follow the instructions below to install the necessary requirements:

1. Ensure that Python version 3.8 is installed on your system.
2. Download or clone this repository to your local machine.
3. Navigate to the project directory using **'cd \<folder>'** command.
4. Run the command **'pip install -r requirements.txt'** to install all the necessary packages.
   
## Usage

To use this repository:

1. **Generate Graphs:**
   Run `utils.py` to produce the visualizations used in the paper.

2. **Re-run Experiments:**
   To repeat a specific experiment, run its corresponding module. For example:

   * To re-run the **LISTA** experiment, execute `unfolding_ista.py`.
3. To run the Yale experiment run the notebook under yale_experiment folder while inserting the data_images in the appropriate path.

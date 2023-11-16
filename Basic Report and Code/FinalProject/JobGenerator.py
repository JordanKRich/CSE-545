import random
import os
import subprocess
from PySide6.QtWidgets import QApplication, QLabel, QPushButton, QGridLayout, QWidget, QLineEdit, QSpinBox
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
# Build the full path to the 'jobsequencing.py' file
filename = os.path.join(script_dir, "JobSequencing.py")

def generate_job_shop_scheduling(num_jobs, num_machines):
    jobs = []
    
    for _ in range(num_jobs):
        operations = []
        
        # Shuffle machine order for each job to ensure randomness
        machines = list(range(num_machines))
        random.shuffle(machines)
        
        for machine in machines:
            processing_time = random.randint(1, 100)  # you can adjust this range
            operations.extend([machine, processing_time])
        
        jobs.append(operations)
    
    return jobs

def run_scripts():
    # Get values from input fields
    num_jobs = num_jobs_input.value()
    num_machines = num_machines_input.value()
    instance_name = instance_name_input.text()

    # Generate the job shop scheduling instance
    jobs = generate_job_shop_scheduling(num_jobs, num_machines)

    # Build the full path to the 'generated_instance.txt' file
    generated_instance_file = os.path.join(script_dir, "generated_instance.txt")

    # Open a text file for writing
    with open(generated_instance_file, "w") as f:
        # Write the name to the file
        f.write("#############################\n")
        f.write(instance_name + "\n")
        f.write("#############################\n")
        # Write Description to file
        description = f"Randomly generated {num_jobs}x{num_machines} instance"
        f.write(description + "\n")
        # Write the number of jobs and machines to the file
        f.write(f"{num_jobs} {num_machines}\n")
        
        # Write the problem instance to the file
        for job in jobs:
            f.write(" ".join(map(str, job)) + "\n")

    print("Instance saved to 'generated_instance.txt'")

    # Run the second Python script
    subprocess.run(["python", filename], check=True)

# Create the Qt Application
app = QApplication(sys.argv)

num_jobs_input = QSpinBox()
num_jobs_input.setMinimum(1)
num_jobs_input.setValue(10)
num_jobs_input.setMaximum(1000)

num_machines_input = QSpinBox()
num_machines_input.setMinimum(1)
num_machines_input.setValue(2)
num_machines_input.setMaximum(1000)

instance_name_input = QLineEdit()
instance_name_input.setText("kentucky_bourbon_factory")

button = QPushButton("Run Scripts")
button.clicked.connect(run_scripts)

layout = QGridLayout()
layout.addWidget(QLabel("Number of Jobs"), 0, 0)
layout.addWidget(num_jobs_input, 0, 1)
layout.addWidget(QLabel("Number of Machines"), 1, 0)
layout.addWidget(num_machines_input, 1, 1)
layout.addWidget(QLabel("Instance Name"), 3, 0)
layout.addWidget(instance_name_input, 3, 1)
layout.addWidget(button, 4, 0, 1, 2)

window = QWidget()
window.setLayout(layout)
window.show()

sys.exit(app.exec_())

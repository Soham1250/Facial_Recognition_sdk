import tkinter as tk
from tkinter import ttk
import mysql.connector  # type: ignore
from utils import DB_HOST, DB_NAME, DB_USER, DB_PASS, DB_PORT

def connect_db():
    conn = mysql.connector.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
        port=DB_PORT
    )
    return conn

# List of employee names (hardcoded)
employee_names = ["Ellaiah Gangadhari",
"Gautam Binniwale",
"Kishor B. Patil",
"Krutik Mandre",
"Krishna Naik",
"M Jayadevi",
"Mahesh S. Bhoop",
"Mannu Vishwakarma",
"Manoj  Yadav",
"Pritesh Mistry",
"Priti Gaikwad",
"Roopnarayan Gupta",
"Sachin Patil",
"Sagar  Tondvalkar",
"Sandesh Kurtdkar",
"Shyamal Mishra",
"Sonal Mayekar",
"Sushil  Khetre",
"Vaibhav Pawar",
"Vikrant Sawant",
"Omkar Nikam",
"Soham Pansare"]

# Function to handle the submit action
def submit_selection():
    selected_employee = selected_name.get()
    print(f"Selected Employee: {selected_employee}")  # Print the selected name to the terminal
    
    # Connect to the database
    conn = connect_db()
    cursor = conn.cursor()
    
    # Update the frequency in the database for the selected employee
    query = "UPDATE employee_selection SET Frequency = Frequency + 1 WHERE Name = %s"
    cursor.execute(query, (selected_employee,))
    conn.commit()  # Commit the transaction to save changes
    
    # Close the database connection
    cursor.close()
    conn.close()
    
    # Close the Tkinter window and exit the program
    app.destroy()
    exit()

# Set up the main Tkinter window
app = tk.Tk()
app.title("Employee Selection")
app.geometry("600x400")  # Set the window size to 600x400

# Label for the dropdown instruction
label = tk.Label(app, text="Select your name", font=("Helvetica", 16))
label.pack(pady=(40, 10))

# Variable to hold the selected employee's name
selected_name = tk.StringVar()
selected_name.set("Your name")  # Placeholder text as shown in the design

# Create the dropdown menu (OptionMenu) for employee selection
dropdown = ttk.OptionMenu(app, selected_name, *employee_names)
dropdown.config(width=20)  # Set width for a more styled dropdown
dropdown.pack(pady=10)

# Add the Select button, centered below the dropdown
submit_button = tk.Button(app, text="Submit", command=submit_selection, font=("Helvetica", 12), width=10)
submit_button.pack(pady=30)

# Run the Tkinter main loop
app.mainloop()

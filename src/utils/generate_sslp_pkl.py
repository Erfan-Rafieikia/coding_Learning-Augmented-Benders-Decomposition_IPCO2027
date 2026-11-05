from pysmps import smps_loader
#r"C:\Programming\Research Projects\Benders_Integration_with_ML\DATA\SSLP\sslp_5_25_50"
data = smps_loader.load_smps(r"C:\Programming\Research Projects\Benders_Integration_with_ML\DATA\SSLP\sslp_5_25_50")  # directory with .cor, .tim, .sto
c, A, b = data["core"]        # Objective coeffs, constraint matrix, RHS from the core file
stage_info = data["time"]     # Dictionary: stage assignments for rows and cols (from .tim)
scenarios = data["sto"]       # Stochastic info: scenario data from .sto file

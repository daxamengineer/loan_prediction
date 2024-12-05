import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('loan_approval_dataset.csv')

# Map loan status to encoded values
df['loan_approved_encoded'] = df[' loan_status'].map({' Approved': 1, ' Rejected': 0})

# Plot Loan Approval vs. CIBIL Score
plt.figure(figsize=(8, 6))
plt.scatter(
    df[' cibil_score'], 
    df['loan_approved_encoded'], 
    c=df['loan_approved_encoded'], 
    cmap='bwr', 
    edgecolor='k', 
    alpha=0.7
)
plt.title('Loan Approval vs. CIBIL Score', fontsize=14)
plt.xlabel('CIBIL Score', fontsize=12)
plt.ylabel('Loan Approved (1) / Rejected (0)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.colorbar(label='Loan Status')
plt.show()

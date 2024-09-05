import pandas as pd

# The issue may stem from extra or missing entries in the lists. Let's check the length of each list first.

compounds = [
    "1,3,5-Trimethylbenzene", "1-(2-decyldodecyl)-2,4-dimethylcyclopentane", 
    "11-(2,2-dimethylpropyl)henicosane", "11-butyldocosane", 
    "11-pentylhenicosane", "2,2,4-Trimethylpentane", "2,2,4-trimethylpentane", 
    "2,2-Dimethylbutane", "2,2-dimethylbutane", "2-Methylpentane", 
    "2-methylbut-2-ene", "2-methylbutane", "2-methylhexane", "2-methylpentane", 
    "2-methyltricosane", "3-ethyltetracosane", "3-methylpentane", 
    "5,14-dibutyloctadecane", "6,11-dipentylhexadecane", 
    "9-ethyl-9-heptyloctadecane", "Anthracene", "Butylbenzene", "Cumene", 
    "Decane", "Dodecane", "Ethylbenzene", 
    "Ethylcyclohexane", "Heptane", "Hexadecane", "Mesitylene", 
    "Methylcyclohexane", "Naphthalene", "Octane", "Pentadecane", 
    "Tetradecane", "Toluene", "Tridecane", "Undecane", "cyclohexane", 
    "cyclohexene", "cyclooctane", "cyclooctatetraene", "cyclopentane", 
    "cyclopentene", "decane", "dodecane", "ethylcyclohexane", 
    "heptadecane", "heptane", "hexadecane", "hexane", "icosane", 
    "m-Xylene", "methylcyclohexane", "methylcyclopentane", "o-Xylene", 
    "octadecane", "octane", "p-Xylene", "pentadecane", "pentane", 
    "tetracosane", "tetradecane", "triacontane", "tridecane", "undecane"
]

classes = [
    "Single Aromatics", "Strongly Branched Isoalkanes", "Strongly Branched Isoalkanes", 
    "Strongly Branched Isoalkanes", "Strongly Branched Isoalkanes", 
    "Strongly Branched Isoalkanes", "Strongly Branched Isoalkanes", 
    "Strongly Branched Isoalkanes", "Strongly Branched Isoalkanes", 
    "Weakly Branched Isoalkanes", "Alkenes", "n-Alkanes", 
    "Weakly Branched Isoalkanes", "Weakly Branched Isoalkanes", 
    "Strongly Branched Isoalkanes", "Strongly Branched Isoalkanes", 
    "Weakly Branched Isoalkanes", "Strongly Branched Isoalkanes", 
    "Strongly Branched Isoalkanes", "Strongly Branched Isoalkanes", 
    "Poly Aromatics", "Single Aromatics", "Single Aromatics", 
    "n-Alkanes", "n-Alkanes", "Single Aromatics", 
    "n-Alkanes", "n-Alkanes", "n-Alkanes", "Single Aromatics", 
    "Weakly Branched Isoalkanes", "Poly Aromatics", "n-Alkanes", 
    "n-Alkanes", "n-Alkanes", "Single Aromatics", "n-Alkanes", 
    "n-Alkanes", "n-Alkanes", "Alkenes", "n-Alkanes", "Alkenes", 
    "n-Alkanes", "Alkenes", "n-Alkanes", "n-Alkanes", "n-Alkanes", 
    "n-Alkanes", "n-Alkanes", "n-Alkanes", "n-Alkanes", 
    "Single Aromatics", "Weakly Branched Isoalkanes", "Weakly Branched Isoalkanes", 
    "Single Aromatics", "n-Alkanes", "n-Alkanes", "Single Aromatics", 
    "n-Alkanes", "n-Alkanes", "n-Alkanes", "n-Alkanes", "n-Alkanes", 
    "n-Alkanes", "n-Alkanes"
]

# Check the lengths of both lists
len(compounds), len(classes)


# Identify the missing compound and add it to the appropriate class

# Assuming "cyclohexane" was classified as "n-Alkanes"
compounds.append("cyclohexane")
classes.append("n-Alkanes")

# Now the lengths should match, so let's generate the CSV again

# Create a DataFrame
df_corrected = pd.DataFrame({
    "Compound": compounds,
    "Class": classes
})

# Save DataFrame as CSV file
corrected_csv_path = "NMR_compounds_corrected.csv"
df_corrected.to_csv(corrected_csv_path, index=False)

corrected_csv_path

import json
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import json
from rdkit.Chem import Draw
import matplotlib.pyplot as plt

# Load data
with open('SurfTTrainingData.json') as f:
    data = json.load(f)
#load each molecule and draw image
for i in data:
    Inchi = i['INChI']
    mol = Chem.MolFromInchi(Inchi)
    img = Draw.MolToImage(mol)
    #smiles to name
    Names= (i['MolName']).replace(' ','_').replace('/','_').replace('-','_')
    smiles = Chem.MolToSmiles(mol)
    img.save(f'Images/{Names}.png')
    print(smiles, Inchi)
#take a folder of images and make a  collage
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

# Load images
images = [Image.open(f'Images/{i["MolName"].replace(" ","_").replace("/","_").replace("-","_")}.png') for i in data]

# Calculate the number of rows and columns
n = len(images)
rows = int(n**0.33)
cols = (n + rows - 1) // rows

# Calculate the size of the collage
widths, heights = zip(*(i.size for i in images))
max_width = max(widths)
max_height = max(heights)
collage_width = cols * max_width
collage_height = rows * max_height

# Create a new image
collage = Image.new('RGB', (collage_width, collage_height), (255, 255, 255))
# Paste the images into the collage
for i in range(n):
    x = max_width * (i % cols)
    y = max_height * (i // cols)
    collage.paste(images[i], (x, y))
#add names to each image
draw = ImageDraw.Draw(collage)
#load font size 16
font = ImageFont.truetype("arial.ttf", 28)

#fontsize
font_size = 12
for i in range(n):
    x = max_width * (i % cols)
    y = max_height * (i // cols)
    #first 20 characters
    draw.text((x, y), data[i]['MolName'][:20], fill=(0, 0, 0), font=font)
#save
collage.save('Images/Collage.png')



import json
with open('NMRData.json', 'r') as infile:
    NMRData = json.load(infile)

#add new data to the NMRData manually, Make a function, require inchi,inchikey, molname
def AddData(InChI,InChIKey,MolName):
    NMRData.append({'INChI':InChI,'INChI key':InChIKey,'MolName':MolName,'13C_shift':[],'1H_shift':[]})
    with open('NMRData.json', 'w') as outfile:
        json.dump(NMRData, outfile)

#add tridecane
AddData('InChI=1S/C13H28/c1-3-5-7-9-11-13-12-10-8-6-4-2/h3-13H2,1-2H3','XEFQLINVKZKXJR-UHFFFAOYSA-N','tridecane')
#add tetradecane
AddData('InChI=1S/C14H30/c1-3-5-7-9-11-13-14-12-10-8-6-4-2/h3-14H2,1-2H3','QGJUPKKBFWGKPL-UHFFFAOYSA-N','tetradecane')
#add pentadecane
AddData('InChI=1S/C15H32/c1-3-5-7-9-11-13-15-14-12-10-8-6-4-2/h3-15H2,1-2H3','QGJUPKKBFWGKPL-UHFFFAOYSA-N','pentadecane')
#add hexadecane
AddData('InChI=1S/C16H34/c1-3-5-7-9-11-13-15-16-14-12-10-8-6-4-2/h3-16H2,1-2H3','QGJUPKKBFWGKPL-UHFFFAOYSA-N','hexadecane')

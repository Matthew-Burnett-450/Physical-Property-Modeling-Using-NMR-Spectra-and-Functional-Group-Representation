import pyautogui as pg
import numpy as np
import time
import json

#load NMRData\molecules.csv 
"""# Name	Structure
Hexane	CCCCCC
Heptane	CCCCCCC
Octane	CCCCCCCC
Nonane	CCCCCCCCC
Decane	CCCCCCCCCC"""
HydrocarbonData = np.loadtxt('NMRData\molecules.csv',delimiter='\t',dtype='str',skiprows=1)
Names = HydrocarbonData[:,0]
Smiles = HydrocarbonData[:,1]

pg.sleep(3)

for i,name in enumerate(Names):

    

    print(name)

    print(len(HydrocarbonData)- i)
    #press control n
    pg.hotkey('ctrl','n')
    pg.sleep(0.1)
    #ctrl a wait
    pg.hotkey('ctrl','a')
    pg.sleep(.5)
    #type name
    pg.typewrite(name)
    pg.sleep(0.1)
    #press enter
    pg.press('enter')
    pg.sleep(0.5)
    try:
        if pg.locateOnScreen('ThermoData\Scraper\DrawData.png',confidence=0.8)!=None:
            #press escape
            pg.press('esc')
            pg.sleep(0.5)
            pg.press('esc')
            pg.sleep(0.5)
            
            continue
    except:
        pass
    try:
        if pg.locateOnScreen('ThermoData\Scraper\ErrorBox.png',confidence=0.8)!=None:
            #press escape
            pg.press('esc')
            pg.sleep(0.1)
            pg.press('esc')

            continue
    except:
        pass
    #press control g
    pg.hotkey('ctrl','g')
    pg.sleep(0.1)

    Eval=False
    while Eval==False:
        print('waiting')
        try:
            try:
                if pg.locateOnScreen('ThermoData\Scraper\EvaluatedData.png',confidence=0.5)!=None:
                    Eval=True
                    print('Evaluated')

            except:
                pass
                #if pg.locateOnScreen('Scraper\EvaluatedData2.png',confidence=0.6)!=None:
                #    Eval=True
                #    print('Evaluated')
            pg.sleep(0.1)
        except:
            pass

    pg.sleep(1)

    Eval=False

    while Eval==False:
        pg.hotkey('ctrl','s',pauses=0.1)
        pg.sleep(.5)
        try:
            if pg.locateOnScreen('ThermoData\Scraper\EvalReact.png',confidence=0.8)!=None:
                #press escape
                pg.press('right')
                pg.sleep(0.1)
                pg.press('enter')
                pg.sleep(0.1)
        except:
            pass
        try:
            if pg.locateOnScreen('ThermoData\Scraper\SaveBoxOpen.png',confidence=0.7)!=None:
                Eval=True
        except:
            pass
    Eval=False

    #type name
    print(name)

    pg.typewrite(name,interval=.05)
    pg.sleep(0.5)
    #enter
    pg.press('enter')
    pg.sleep(1)


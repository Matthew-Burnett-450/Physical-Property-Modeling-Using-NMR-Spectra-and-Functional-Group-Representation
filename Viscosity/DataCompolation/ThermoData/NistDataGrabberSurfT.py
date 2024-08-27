import numpy as np
import ThermoMlReader as tml






class TargetConstructor():
    def __init__(self,StateEquations):
        self.StateEquations=StateEquations
    def GenerateTargets(self,FilePath,FileName,num=50,state=None):

    
        tml_parser = tml.ThermoMLParser(FilePath)

        tml_parser.extract_properties()
        tml_parser.extract_equation_details()
        Properties=tml_parser.get_properties()
        
        #if FileName not in Properties['property_names']:
        #    return [],[],False
        if FileName not in Properties['property_names']:
            return [],[],False
        if state!=None:
            if state not in Properties['property_phase'][Properties['property_names'].index(FileName)]:
                return [],[],False

        idx = np.where(np.isin(Properties['equation_names'], list(self.StateEquations.keys())))[0]
        EqName = [Properties['equation_names'][i] for i in idx]
        idx=int(idx[-1])
        #print(EqName)
        #dx=Properties['equation_names'].index(EqName[0])

        Params=Properties['equation_parameters'][idx]
        VarRange=Properties['variable_ranges'][idx]


        # Generate a linspace for each variable range and stack them horizontally.
        T = np.column_stack([np.linspace(var_min, var_max, num) for var_min, var_max in VarRange])

        StateEquation=self.StateEquations[EqName[-1]]
        Y=StateEquation.run(T,Params)

        #compute A and B


        self.x=T.flatten().tolist()
        self.Y=Y.flatten().tolist()
        
        print(len(self.x),len(self.Y))  
        return self.x,self.Y,EqName[-1] #return the temperature, surface tension, and critical temperature



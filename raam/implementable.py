"""
Tools for formulating interpretable/implementable MDPs as mathematical optimization
problems (MILP). This solves the models as described in:
Petrik, M., & Luss, R. (2016). Interpretable Policies for Dynamic Product Recommendations. In Uncertainty in Artificial Intelligence (UAI).

Usage:

    - Use get_milp_model to create "model.mod" MILP model to run with the data output.
    - create_opl_data can be used to transform an interpretable MDP data to a MILP
    - solve_opl solves the model

"""

import numpy as np
import subprocess
import json
import io

#%% Create OPL data file

import random

def create_opl_data(name, mdp, p0, observations, discount, filename="milp_output.dat",
                    action_decomposition = None):
    """
    Creates an OPL representation of the interpretable MDP. OPL is a modelling
    language included with CPLEX optimizer. This representation can be used
    to conveniently solve LP and MILP formulations of MDPs
    
    Parameters
    ----------
    name : string
        Name of the problem
    mdp : craam.MDP
        MDP specification with no robustness
    observations : array-like
        Observation index for each state
    discount : float 
        Discount factor
    filename : string, optional
        Name of the opl data output file
        If None, the outputs the string
    action_decomposition : list of lists, optional
        If actions can be decomposed then for each action it lists the indexes
        of decomposed actions. 
        
    Returns
    -------
    ident : string
        Identifier of the problem (to check that the solution is for the same)
    string_rep : string
        Only if provided with filename=None, also returns OPL string representation
    """

    ident = str(random.randint(0,1e6))

    if filename is not None:
        f = open(filename,"w")
    else:
        f = io.StringIO()
    
    try: 
        f.write('problemName="'+name+'";\n')
        f.write('discount='+str(discount)+';\n')
        f.write('upper_bound='+str(1/discount)+';\n')
        f.write('generatedID="' + ident + '";\n')
        
        f.write('initial = {') 
        first = True
        for s in range(mdp.state_count()):
            if p0[s] > 0:
                if first:
                    f.write('<')
                    first = False
                else:
                    f.write(',<')
                f.write(str(s) + ',' + str(p0[s]) + '>')
        f.write('};\n')
        
        f.write('states = {')    
        first = True
        for s in range(mdp.state_count()):            
            if first:                        
                first = False
            else:
                f.write(',')
            f.write(str(s));    
        f.write('};\n')
        
        f.write('samples = {')    
        first = True
        for s in range(mdp.state_count()):
            for a in range(mdp.action_count(s)):
                for ts, prob, rew in zip(mdp.get_toids(s,a,0), mdp.get_probabilities(s,a,0), mdp.get_rewards(s,a,0)):
                    if first:
                        f.write('<')
                        first = False
                    else:
                        f.write(',<')
                    f.write(','.join((str(s),str(a),str(ts),str(prob),str(rew))) + '>')    
        f.write('};\n')
        
        f.write('observation_states = {')            
        first = True
        for s in range(mdp.state_count()):
            if first:
                f.write('<')
                first = False
            else:
                f.write(',<')
            f.write(str(observations[s]) + ',' + str(s) + '>')
        f.write('};\n')
        
        if action_decomposition is not None:
            assert len(action_decomposition) == mdp.action_count(1)
            f.write('decomposed_actions = {')            
            first = True
            for i,ad in enumerate(action_decomposition):
                if first:
                    f.write('<')
                    first = False
                else:
                    f.write(',<')                
                f.write(str(i)+',')
                f.write(','.join(str(a) for a in ad))
                f.write('>')
            f.write('};\n')
            
        if filename is not None:
            return ident
        else:
            return ident, f.getvalue()

    finally:
        f.close()


def solve_opl(model='milp.mod', data="milp_output.dat", result="solution.json", \
                ident = None, oplrun_exec="oplrun", verbose=False):
    """
    Solves the OPL formulation constructed by create_opl_data.
    All files must be in the same directory.
    
    Parameters
    ----------
    model : string, optional
        Name of the model to run
    data : string
        Name of the data file
    result : string, optional
        Name of the output file
    ident : string, optional
        Solution identifier to make sure that the solution is for the correct
        problem
    oplrun_exec : string, optional 
        Path to the oplrun executable "CPLEX"
        
    Note
    ----
    One may also need to set the path to the OPL libraries (e,g., libicuuc.so), 
    by using:
        oplrun_exec = 'export LD_LIBRARY_PATH=/opt/ibm/ILOG/CPLEX_Studio_Community1263/opl/bin/x86-64_linux/; /opt/ibm/ILOG/CPLEX_Studio_Community1263/opl/bin/x86-64_linux/oplrun'
        
    The method requires a shell to be present

    
        
    Returns
    -------
    obj : float
        Objective value
    psi : array
        Interpretable policy, action index for each observation
    """
    try:
        command = [oplrun_exec, model, data]
        if verbose:
            print("Executing:", " ".join(command))
        
        stdout = subprocess.check_output(" ".join(command), shell=True)
        
        
        if verbose:
            print("Output")
            print(stdout.decode('utf-8'))
            
    except subprocess.CalledProcessError as e:
        print('OPL failed with:')
        print(e.output.decode('utf-8'))
        raise e

    with open("solution.json", mode='r') as fileinput:
        datainput = fileinput.read()
 
        d = json.JSONDecoder().decode(datainput)
     
        if ident is not None:
            ident_sol = d['GeneratedId']     
        
            assert ident == ident_sol, "Solution identifier does not match problem identifier" 
     
        obj = d['Objective']
        oc = d['ObservCount']
        ac = d['ActionCount']
        psi = d['psi']
        
        psi = np.reshape(psi,(oc,ac))
        psi = psi.argmax(1)
        
        return obj, psi
        
def get_milp_model():
    """
    Returns a string definition a MILP model that work with the generated OPL data.
    
    Save the output as model.mod
    """
    return \
    r"""
    /********************************************************************
    *  MILP formulation for interpretable MDPs
    *********************************************************************/
    
    execute {cplex.tilim = 180;};
    
    // Used to identify the problem
    string problemName = ...; 
    // Used to identify the problem run
    string generatedID = ...; 
    
    float upper_bound = ...;
    
    // The discount factor 
    float discount = ...;
    
    // Sample definition
    tuple Sample {
        int sourceid;
        int action;
        int targetid;
        float probability;
        float reward;
    };
    
    tuple Initial{
        int stateid;
        float probability;
    };
    
    tuple Observation {
        int observid;
        int stateid;
    };
    
    {Initial} initial = ...;
    {Sample} samples = ...;
    {Observation} observation_states = ...;
    
    {int} states = ...; //{s.sourceid | s in samples};
    {int} actions = {s.action | s in samples};
    {int} observations = {os.observid | os in observation_states};
    
    // Occupancy frequency
    dvar float+ u[states][actions];
    
    // Interpretable policy
    //dvar int psi[observations][actions] in 0..1;
    dvar boolean psi[observations][actions];
    
    
    dexpr float reward[s in states][a in actions] = 
        sum(ss in samples : ss.sourceid == s && ss.action == a) ss.reward * ss.probability;
    dexpr float d[s in states] = sum(a in actions) u[s][a];
    dexpr float initdist[s in states] = sum(ss in initial : ss.stateid == s) ss.probability;
    dexpr float objective = sum(s in states, a in actions) u[s][a] * reward[s][a];
            
    maximize objective;
    
    subject to {
        forall(s in states){
            d[s] == initdist[s] + sum(ss in samples : ss.targetid == s) 
                    (discount * u[ss.sourceid][ss.action] * ss.probability);
        };
    
        forall(o in observations){
            forall(os in observation_states : os.observid == o){
                forall(a in actions){
                    u[os.stateid][a] <= upper_bound * psi[o][a];
                    upper_bound*(psi[o][a] -1) + d[os.stateid] <= u[os.stateid][a];
                };
            };
        };
    
        forall(o in observations){
            sum(a in actions) psi[o][a] == 1;
        };
    };
        
    execute{
        var result = new IloOplOutputFile("solution.json");
        var counter = 0;
        result.write("{");
        
        result.write("\"ProblemName\" : \"" + problemName + "\",\n");
        result.write("\"GeneratedId\" : \"" + generatedID + "\",\n");
        
        result.write("\"Algorithm\" : ");
        result.write("\"MILP\",\n");
        
        result.write("\"Objective\" : ");
        var obj = "" + objective;
        //prevent a trailing dot
        if(obj.indexOf(".") == obj.length-1) obj = obj + "0";
        result.write(obj);
        result.write(",\n");
    
        result.write("\"StateCount\" : ");
        result.write(thisOplModel.states.size);
        result.write(",\n");
    
        result.write("\"ObservCount\" : ");
        result.write(thisOplModel.observations.size);
        result.write(",\n");
    
        result.write("\"ActionCount\" : ");
        result.write(thisOplModel.actions.size);
        result.write(",\n");
    
        result.write("\"psi\" : [");
        for(var o in thisOplModel.observations){
            for(var a in thisOplModel.actions){
                if(counter > 0) result.write(",");
                var val = "" + thisOplModel.psi[o][a];
                //prevent trailing dot
                if(val.indexOf(".") == val.length - 1) val = val + "0";
                result.write(val);
                counter = counter + 1;
            }
        }
        result.write("]");
        result.write("}\n");
        result.close();
    } 

    """

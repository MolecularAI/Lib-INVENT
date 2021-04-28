Lib-INVENT Tutorials
=======================================================================================================================================
Four tutorials are provided along with the project. The purpose of these tutorials is to illustrate the process of the 
assembly of a configuration JSON file and to motivate various parameter choices the user may want to make.

The first tutorial focuses on Teacher's Forcing and demonstrates how a prior model can be trained.

The three remaining tutorials focus on reinforcement learning and demonstrate the usage of different scoring components
and parameters which have been used in the publication. They are intended to be read in order as the depth of explanations
gradually decreases; starting from more detailed motivation for parameter choices and explanations of alternative options
for custom parameters. The subsequent tutorials comment primarily on the parts of the JSON which differ from the previous 
tutorials.

Two examples of results analysis are further provided in the first and second RL tutorials. The purpose of these is to
illustrate the interpretation of the provided tensorboard plots and highlight important features the user might want to
consider when analysing the results of a completed run.

### General usage
The tutorials are provided in the form of jupyter notebooks. To run them from command line:

`$ conda activate lib-invent` \
`$ cd <project_directory>/tutorial`\
`$ jupyter notebook`

If a browser window does not open automatically, copy the link which appears in your console and open it.

In order to execute the notebooks, path placeholders in these need to be replaced with actual paths on the local machine.

Finally, note that the notebooks are intended as an illustration of the assembly process and to provide explanations of the various
parameters necessary to execute a Lib-INVENT run. In practice, it is also possible to assemble the JSON file directly in
the local directory of the user or to simply edit a previously used JSON instead of creating it in the jupyter notebook.
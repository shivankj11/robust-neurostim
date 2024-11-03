# robust-neurostim

#### Design of Electrode Placement for Neurostimulation to be Robust with Respect to Patient Physiology

### Deck
[Google Slides Link](https://docs.google.com/presentation/d/1nWs95Ar3eWqMNnfPgwMIIZhzAtu8QDRs/edit?usp=sharing&ouid=117710712997531836597&rtpof=true&sd=true)

### Usage
`pip install robust-neurostim`

### Abstract
This research aimed to  investigate whether the optimization for electrode placement and current passed through electrodes to generate an electric field in the brain could be made robust. The optimization approach used in this study focused on minimizing the current outside of the region being targeted for stimulation with respect to uncertainty in the conductivities of each section of a spherical head model in the worst case. The robust approach used sequential rounds of particle swarm optimization to find worst case conductivity parameters for an electrode setup and convex optimization to find a new electrode setup for new conductivity parameters. The study found that the robust optimization approach was beneficial in worst-case scenarios, reducing the electric field outside of the target region by 12.5% compared to the original setup. However, the robust approach performed worse than the original approach in the average case. This work suggests that this robust electrode design scheme for stimulation experimentation and medical treatments may minimize the neurons outside the targeted region that may be activated, in turn potentially reducing side-effects. Further research is required to validate these findings, compare other robust formulations, and determine how other uncertainties affect the robust optimization performance. 


### Summary
This research aimed to investigate whether the optimization for electrode placement and 
current passed through those electrodes to generate an electric field in the brain could be made 
robust and whether this provided any benefits. This is relevant in designing electrode placement 
schema for stimulation experimentation and medical treatments since a robust placement scheme 
may minimize the neurons outside the targeted region for stimulation that are activated, in turn 
reducing potential side-effects.

The identified sources of uncertainty in the original model were the radii of the different 
sections of the head and the conductivities of each section. This study focused on uncertainty in 
the conductivities of each section of the head model since the radii can be determined accurately 
via MRI scans. This study built off work that developed a model and optimization scheme to 
place electrodes on the scalp and pass current through those electrodes to generate a desired 
electric field in some 'focus' regions in the brain while minimizing the field in other 'cancel' 
regions in the brain.

The investigated robust approach iteratively maximized the error on the electrode setup 
chosen by varying the conductivity parameters within some tolerance and then found a new 
electrode placement for these new parameters. At each step, particle swarm optimization was 
used to find new values for conductivities and the original convex optimization technique was 
used to find the new electrode setup. The findings of the study revealed that there are benefits of 
robust optimization in a worst-case scenario. The electrode setup designed by the robust 
optimization had 12.5% less electric field in the ‘cancel' region than the original electrode setup 
in the worst-case scenario. There was, however, no benefit of using robust optimization in the 
average case and the findings suggest that the particular robust optimization used in this study 
may be worse in the average case.

The results from this study show that given uncertainty in physical parameters in the head 
model, such as conductivities and depths of the layers of the head, there is a benefit to using the 
investigated robust optimization method in the worst case scenario but not in the average case. 
This may help researchers design safer electrode stimulation studies. Further research is required 
to validate these findings, compare other robust formulations, and determine how other 
uncertainties affect the robust optimization performance. 

# UVmodeldisk
This code uses KinMSpy for producing the rotating disk model and uses Galario to convert the images into visibilities.
The purpose of this code is to carry out a full posterior study for the disk model parameters, evaluating the model likelihood directly on the visibility data.

To prepare the visibilities for UV model fitting, follow the steps below, in CASA.

```
split('vis.ms',field='f',outputvis='vis_binned.ms',spw='sp',timebin='30s')
statwt('vis_binned.ms')
```

WHEN USING combinespws=True, have to combine WITHOUT selecting channels, full spws!!  averaging is fine, selecting channels is NOT!

average a few channels together to speed up the modeling:
```mstransform('vis_binned.ms','vis_binned_comb_fullspws.ms',combinespws=True,datacolumn='data',chanaverage=True,chanbin=5,spw='sp')``` 

and only select the channels for fitting, after averaging:
```mstransform('vis_binned_comb_fullspws.ms','vis_binned_comb.ms',spw='ch',datacolumn='data')``` 

Then, in this order, :
```
initweights('vis_binned_comb.ms',wtmode='weight',dowtsp=True)
exportuvfits('vis_binned_comb.ms','vis_binned_comb.uvfits')
```
Potentially run ```statwt``` again at this point, even though it should not be necessary.

Then use either emcee, or pymultinest to run the modeling, through the same class structure defined in UVmodeldisk.py

See Class_testing_script_multinest.py and Class_testing_script_emcee.py for example scripts!

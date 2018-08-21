# uvmodeldisk
#To prepare the visibilities for UV model fitting, follow the steps below, in CASA.

```
split('vis.ms',field='f',outputvis='vis_binned.ms',spw='sp',timebin='30s')
statwt('vis_binned.ms')
```

#WHEN USING combinespws=True, have to combine WITHOUT selecting channels, full spws!!  averaging is fine, selecting channels is NOT!

```mstransform('vis_binned.ms','vis_binned_comb_fullspws.ms',combinespws=True,datacolumn='data',chanaverage=True,chanbin=5,spw='sp')``` 

#average a few channels together to speed up the modeling

```mstransform('vis_binned_comb_fullspws.ms','vis_binned_comb.ms',spw='ch',datacolumn='data')``` 

#only select the channels for fitting, after averaging

```initweights('vis_binned_comb.ms',wtmode='weight',dowtsp=True)```

```exportuvfits('vis_binned_comb.ms','vis_binned_comb.uvfits')```

#Then use either emcee, or pymultinest to run the modeling using the scripts included.

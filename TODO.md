## Config file:

### Remove:
`valid_time` in `valid` section
`callback` in `train` section
`warmup_epochs` in `train` section

### Add:
`optimizer` in `train`
> `SGD`, `RMSProm`, `Adam`

`lr_scheduler` in `train`
> `None`, `CDR`, `ExpD`
> as object with hyper params as attributs

### Which optimizer
5 test with each and compare f1-scores
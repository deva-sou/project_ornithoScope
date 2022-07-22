## Config file:

### Remove:
`valid time` in `valid` section
`callback` in `train` section
`warmup_epochs` in `train` section

### Add:
`optimizer` in `train`
> `SGD`, `RMSProm`, `Adam`

`lr_scheduler` in `train`
> `None`, `CDR`, `ExpD`
> as object with hyper params as attributs

Faire en sorte de ne pas ré-entrainer les poids afin de voir vraiment si l'algo est plus efficace que celui de deva
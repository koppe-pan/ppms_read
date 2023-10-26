# ppms_read
## usage
### `extract_RT`
```
ppms = PPMS(path)
ppms.extract_RT(bridge=1, T=(2,4), iloc=(1334, 1994))
ppms.mk_fig()
```
### `extract_Hc`
```
ppms = PPMS(path)
ppms.extract_Hc(bridge=1, I=(0.9,1.1), iloc=(1334, 1994))
ppms.mk_fig()
```
### `extract_Ic`
```
ppms = PPMS(path)
ppms.extract_Ic(bridge=1, iloc=(0, 1334))
ppms.mk_fig()
```

## build
`python setup.py bdist_wheel`

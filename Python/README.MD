# Note
Clone Janus git repo, put its python lib folder `janus` in this dir.

Before run the weight extractors, you need to create the output folder manually.

The filesystem is like:
```
./
 |--janus/
 |-- ...
 |
 |--model_file/
 |  |--vq/
 |
 |--convert_lang.py
 |--convert_vq.py
```

Then copy the path, and paste it into the string variable `output_dir` in both `.py` files.

`python conver_lang.py` and `python conver_vq.py` are ok.
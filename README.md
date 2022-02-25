# word-recognition
Playing around the MJSynth dataset. Currently looking at classification. Training is done in segments of the top `5k, 10k, 15k, ...` most frequent words of the set.

See `gen_tfrecords.py` for the functions used to preprocess the data.

[Dataset](https://www.robots.ox.ac.uk/~vgg/data/text/)

```
@InProceedings{Jaderberg14c,
  author       = "Max Jaderberg and Karen Simonyan and Andrea Vedaldi and Andrew Zisserman",
  title        = "Synthetic Data and Artificial Neural Networks for Natural Scene Text Recognition",
  booktitle    = "Workshop on Deep Learning, NIPS",
  year         = "2014",
}
                

@Article{Jaderberg16,
  author       = "Max Jaderberg and Karen Simonyan and Andrea Vedaldi and Andrew Zisserman",
  title        = "Reading Text in the Wild with Convolutional Neural Networks",
  journal      = "International Journal of Computer Vision",
  number       = "1",
  volume       = "116",
  pages        = "1--20",
  month        = "jan",
  year         = "2016",
}
                
```
